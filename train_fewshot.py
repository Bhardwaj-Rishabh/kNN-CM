import torch
from datasets import load_dataset, load_metric
import numpy as np
from transformers import set_seed, TrainingArguments, AdapterTrainer, EvalPrediction
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, RobertaModelWithHeads
from transformers import TextClassificationPipeline

from os.path import join
from torch.utils.data import DataLoader

import argparse
from transformers.integrations import TensorBoardCallback
from transformers import EarlyStoppingCallback, default_data_collator
import logging
logging.disable(logging.WARNING)

from preprocess import RLDSDataset, DLRSDataset
from sklearn.metrics import precision_recall_fscore_support

from preprocess import SuperGlueDataset

set_seed(1234)

import random
random.seed(4)

SUPERGLUE = {"cb", "rte", "boolq", "wic", "wsc", "copa", "record", "multirc"}


def get_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--epochs', type=int, default=6)
	parser.add_argument('--batchsize', type=int, default=32)
	parser.add_argument('--fp16', type=bool, default="False")
	parser.add_argument('--metric4train', type=str, default="acc", help="one of {macro_f1, acc}, macro_f1 is for ABAS")
	parser.add_argument('--dataset', type=str)
	parser.add_argument('--output_path', type=str)
	parser.add_argument('--save_adapter_path', type=str)
	parser.add_argument('--pad_to_max_length', type=bool, default=True)
	parser.add_argument('--max_seq_length', type=int, default=128)
	parser.add_argument('--dstore_path', type=str)
	parser.add_argument('--layer_id', type=int, default=11)
	parser.add_argument('--device', type=str)

	#####
	parser.add_argument('--lambdas', type=list, default=[1e-3]) #1e-3, 1e-2, 0.05, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
	parser.add_argument('--topn', type=list, default=[1]) #1, 2, 4, 8, 16, 32, 64, 128, 256, 512
	parser.add_argument('--kl_thresholds', type=list, default=[10]) #0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
	parser.add_argument('--fewshot_percent', type=float)
	#####
	
	args = parser.parse_args()

	return args



def encode_batch(batch):
	"""Encodes a batch of input data using the model tokenizer."""
	return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

def anli_encode_batch(batch):
	"""Encodes a batch of input data using the model tokenizer."""

	outputs = []
	for i in range(len(batch["premise"])):
		outputs.append(tokenizer.encode(text=batch["premise"][i],
                            text_pair=batch["hypothesis"][i],
							max_length=300,
							truncation=True,
							padding="max_length",
                            add_special_tokens=True))
	
	return {"input_ids": outputs,
			"labels": batch["labels"]}

def cb_encode_batch(batch):
	"""Encodes a batch of input data using the model tokenizer."""

	outputs = []
	for i in range(len(batch["premise"])):
		outputs.append(tokenizer.encode(text=batch["premise"][i],
                            text_pair=batch["hypothesis"][i],
							max_length=500,
							truncation=True,
							padding="max_length",
                            add_special_tokens=True))
	
	return {"input_ids": outputs,
			"labels": batch["labels"]}

def qasc_preprocess_function(batch):

	results = {
		"input_ids": list(),
		"attention_mask": list(),
		"labels": list(),
	}

	for idx, question in enumerate(batch["question"]):
		choices, choices_labels, answerkeys = batch["choices"][idx]["text"], batch["choices"][idx]["label"], batch["answerKey"][idx]

		for choi_idx, choi in enumerate(choices):

			result = tokenizer(question, choi, max_length=30, truncation=True, padding="max_length")

			label = 1 if choices_labels[choi_idx] in answerkeys else 0

			results["input_ids"].append(result["input_ids"])
			results["attention_mask"].append(result["attention_mask"])
			results["labels"].append(label)
	
	return results


def get_dataset(data_name, tokenizer, args):

	if data_name == "anli":
		num_of_labels = 3
		id_2_label={0:"0", 1:"1", 2:"2"}

		dataset = load_dataset("anli", cache_dir="/data/yingting/Dataset/anli/")
		dataset = dataset.rename_column("label", "labels")
		
		dataset = dataset.map(anli_encode_batch, batched=True, remove_columns=dataset["train_r2"].column_names)
		dataset.set_format(type="torch", columns=["input_ids",  "labels"])
		
		anli_train_dataset = torch.utils.data.ConcatDataset([dataset["train_r1"], dataset["train_r2"], dataset["train_r3"]])
		anli_valid_dataset = torch.utils.data.ConcatDataset([dataset["dev_r1"], dataset["dev_r2"], dataset["dev_r3"]])
		anli_test_dataset = torch.utils.data.ConcatDataset([dataset["test_r1"], dataset["test_r2"], dataset["test_r3"]])

		num_train = len(anli_train_dataset)
		num_picked_train = int(len(anli_train_dataset) * args.fewshot_percent * 0.01)

		# picked_samples = random.sample(range(num_train), num_picked_train)
		picked_samples = range(num_train)[:num_picked_train]

		# breakpoint()

		anli_train_dataset_new = []
		for p_idx in picked_samples:
			anli_train_dataset_new.append(anli_train_dataset[p_idx])

		# breakpoint()

		dataset = {"train" : anli_train_dataset_new,
				"validation" : anli_valid_dataset,
				"test" : anli_test_dataset}
	else:
		raise NotImplementedError

	return dataset, num_of_labels, id_2_label

def compute_accuracy(p: EvalPrediction):
	preds = np.argmax(p.predictions, axis=1)
	return {"accuracy": (preds == p.label_ids).mean()}

def compute_accuracy_test(predictions, references):
	if isinstance(predictions, tuple):
		predictions = predictions[0]
	predictions = np.argmax(predictions, axis=1)
	metric = load_metric("accuracy")
	return metric.compute(predictions=predictions, references=references)

def compute_macro_f1(p: EvalPrediction):
	preds = np.argmax(p.predictions, axis=1)
	p_weighted, r_weighted, f_weighted, support_weighted = precision_recall_fscore_support(p.label_ids, preds, average='macro')
	return {"macro_f1":f_weighted}

def compute_accuracy_knnlm(labels, predictions):
	preds = np.argmax(predictions, axis=-1)
	nums = len(labels)
	count = 0
	for i,(l, p) in enumerate(zip(labels, preds)):
		if (l==p):
			count += 1

	return round(count * 1.0/nums, 4)


if __name__ == "__main__":

	### Args
	args = get_args()

	# rotten_tomatoes: lr=1e-4, epochs=6
	# trec: lr=1e-4, epochs=6
	# anli3: lr=1e-4, epochs=6, batch_size = 32
	# anli1: lr=1e-3, epochs=20
	# anli2: lr=1e-3, epochs=20
	# restaurant: max_length=80(max_text_len is 69), lr=1e-3, epochs=20
	# laptop: max_length=90(max_text_len is 78), lr=1e-3, epochs=20
	# cb: lr=3e-3, epochs=30, batch_size = 4

	log_path = join(args.output_path, "log")
	training_args = TrainingArguments(
		learning_rate=args.lr,
		num_train_epochs=args.epochs,
		per_device_train_batch_size=args.batchsize,
		per_device_eval_batch_size=args.batchsize,
		# per_device_eval_batch_size=1,
		logging_steps=100, #25
		output_dir=args.output_path,
		overwrite_output_dir=True,
		load_best_model_at_end=True,
		metric_for_best_model=args.metric4train,
		# metric_for_best_model="macro_f1",
		evaluation_strategy='steps',
		save_strategy = "steps",
		# max_steps=5000,
		eval_steps=500,  #25
		save_steps=500,
		warmup_steps=5000,
		logging_dir=log_path,
		# The next line is important to ensure the dataset labels are properly passed to the model
		remove_unused_columns=False,
		#changed
		# remove_unused_columns=True,
	)

	#### Dataset
	tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
	dataset, num_of_labels, id_2_label = get_dataset(args.dataset, tokenizer, args)

	### Config
	config = RobertaConfig.from_pretrained(
		"roberta-base",
		num_labels=num_of_labels
	)

	### Model
	model = RobertaModelWithHeads.from_pretrained(
		"roberta-base",
		config=config,
	)
	print("------>>> Trainable params(before freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))

	#changed
	# Add a new adapter
	model.add_adapter(args.dataset)
	
	if args.dataset == "copa":
		# Add a multiple choice head
		model.add_multiple_choice_head(
			args.dataset,
			num_choices=2
		)
	else:
		model.add_classification_head(
			args.dataset,
			num_labels=num_of_labels,
			id2label=id_2_label
		)
	
	# Activate the adapter
	model.train_adapter(args.dataset)

	print("------>>> Trainable params(after freeze):", sum(p.numel() for p in model.parameters() if p.requires_grad))
	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name, param.requires_grad, param.size())

	print(model)

	
	# print(dataset)
	train_dataset = dataset["train"]
	valid_dataset = dataset["validation"]
	test_dataset = dataset["test"]
	data_collator = default_data_collator
	compute_metrics = compute_accuracy

	# breakpoint()


	### Trainer
	trainer = AdapterTrainer(
		model=model,
		args=training_args,
		# train_dataset=dataset["train"],
		# eval_dataset=dataset["validation"],
		#############
		train_dataset=train_dataset,
		eval_dataset=valid_dataset,
		# compute_metrics=dataset.compute_metrics,
		tokenizer=tokenizer,
        # data_collator=dataset.data_collator,
		###############3
		data_collator=data_collator,
		compute_metrics=compute_metrics,
		# compute_metrics=compute_macro_f1,
		callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)]
		# callbacks = [TensorBoardCallback]
	)
	
	trainer.train()
	valid_metric = trainer.evaluate()

	print(valid_metric)
	# print(trainer.predict(dataset["test"]).metrics)
	print(trainer.predict(test_dataset).metrics)
	model.save_adapter(args.save_adapter_path, args.dataset)

	
