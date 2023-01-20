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

SUPERGLUE = {"rte", "boolq", "wic", "wsc"}


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


def get_dataset(data_name, tokenizer, args):

	if data_name == "rotten_tomatoes":

		dataset = load_dataset("rotten_tomatoes", cache_dir="./Dataset/rotten_tomatoes/")
		dataset = dataset.rename_column("label", "labels")

		num_of_labels = 2
		id_2_label={ 0: "👎", 1: "👍"}

		# Encode the input data
		dataset = dataset.map(encode_batch, batched=True)
		# Transform to pytorch tensors and only output the required columns
		dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

	elif data_name == "trec":

		dataset = load_dataset("trec", cache_dir="./Dataset/trec/")
		dataset['train'] = dataset['train'].shuffle(seed=42)
		dataset_tr_valid = dataset['train'].train_test_split(test_size=0.1, shuffle=True, seed=100)
		dataset['train'] = dataset_tr_valid['train']
		dataset['validation'] = dataset_tr_valid['test']
		dataset = dataset.rename_column("coarse_label", "labels")

		num_of_labels = 6
		id_2_label={ 0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}

		# Encode the input data
		dataset = dataset.map(encode_batch, batched=True)
		# Transform to pytorch tensors and only output the required columns
		dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

	elif data_name == "anli3":

		dataset = load_dataset("anli", split=["train_r3","dev_r3","test_r3"], cache_dir="./Dataset/anli/")
		dataset[0] = dataset[0].rename_column("label", "labels")
		dataset[1] = dataset[1].rename_column("label", "labels")
		dataset[2] = dataset[2].rename_column("label", "labels")

		num_of_labels = 3
		id_2_label={0:"0", 1:"1", 2:"2"}

		# Encode the input data
		dataset[0] = dataset[0].map(anli_encode_batch, batched=True, remove_columns=dataset[0].column_names)
		dataset[1] = dataset[1].map(anli_encode_batch, batched=True, remove_columns=dataset[1].column_names)
		dataset[2] = dataset[2].map(anli_encode_batch, batched=True, remove_columns=dataset[2].column_names)
		# Transform to pytorch tensors and only output the required columns
		dataset[0].set_format(type="torch", columns=["input_ids",  "labels"])
		dataset[1].set_format(type="torch", columns=["input_ids",  "labels"])
		dataset[2].set_format(type="torch", columns=["input_ids",  "labels"])

		dataset = {"train" : dataset[0],
				"validation" : dataset[1],
				"test" : dataset[2]}

	elif data_name == "anli1":
		dataset = load_dataset("anli", split=["train_r1","dev_r1","test_r1"], cache_dir="./Dataset/anli1/")
		dataset[0] = dataset[0].rename_column("label", "labels")
		dataset[1] = dataset[1].rename_column("label", "labels")
		dataset[2] = dataset[2].rename_column("label", "labels")

		num_of_labels = 3
		id_2_label={0:"0", 1:"1", 2:"2"}

		# Encode the input data
		dataset[0] = dataset[0].map(anli_encode_batch, batched=True, remove_columns=dataset[0].column_names)
		dataset[1] = dataset[1].map(anli_encode_batch, batched=True, remove_columns=dataset[1].column_names)
		dataset[2] = dataset[2].map(anli_encode_batch, batched=True, remove_columns=dataset[2].column_names)

		# Transform to pytorch tensors and only output the required columns
		dataset[0].set_format(type="torch", columns=["input_ids",  "labels"])
		dataset[1].set_format(type="torch", columns=["input_ids",  "labels"])
		dataset[2].set_format(type="torch", columns=["input_ids",  "labels"])

		dataset = {"train" : dataset[0],
				"validation" : dataset[1],
				"test" : dataset[2]}

	elif data_name == "anli2":
		dataset = load_dataset("anli", split=["train_r2","dev_r2","test_r2"], cache_dir="./Dataset/anli2/")
		dataset[0] = dataset[0].rename_column("label", "labels")
		dataset[1] = dataset[1].rename_column("label", "labels")
		dataset[2] = dataset[2].rename_column("label", "labels")

		num_of_labels = 3
		id_2_label={0:"0", 1:"1", 2:"2"}

		# Encode the input data
		dataset[0] = dataset[0].map(anli_encode_batch, batched=True, remove_columns=dataset[0].column_names)
		dataset[1] = dataset[1].map(anli_encode_batch, batched=True, remove_columns=dataset[1].column_names)
		dataset[2] = dataset[2].map(anli_encode_batch, batched=True, remove_columns=dataset[2].column_names)

		# Transform to pytorch tensors and only output the required columns
		dataset[0].set_format(type="torch", columns=["input_ids",  "labels"])
		dataset[1].set_format(type="torch", columns=["input_ids",  "labels"])
		dataset[2].set_format(type="torch", columns=["input_ids",  "labels"])

		dataset = {"train" : dataset[0],
				"validation" : dataset[1],
				"test" : dataset[2]}

	elif data_name == "anli":
		num_of_labels = 3
		id_2_label={0:"0", 1:"1", 2:"2"}

		dataset = load_dataset("anli", cache_dir="./Dataset/anli/")
		dataset = dataset.rename_column("label", "labels")
		
		dataset = dataset.map(anli_encode_batch, batched=True, remove_columns=dataset["train_r2"].column_names)
		dataset.set_format(type="torch", columns=["input_ids",  "labels"])
		
		anli_train_dataset = torch.utils.data.ConcatDataset([dataset["train_r1"], dataset["train_r2"], dataset["train_r3"]])
		anli_valid_dataset = torch.utils.data.ConcatDataset([dataset["dev_r1"], dataset["dev_r2"], dataset["dev_r3"]])
		anli_test_dataset = torch.utils.data.ConcatDataset([dataset["test_r1"], dataset["test_r2"], dataset["test_r3"]])

		dataset = {"train" : anli_train_dataset,
				"validation" : anli_valid_dataset,
				"test" : anli_test_dataset}
    
	elif data_name == "restaurant":
		data_path = "./Dataset/RLDS"
		r_train_v2 = "Restaurants_Train_v2.csv"
		restaurant_file = join(data_path, r_train_v2)

		num_of_labels = 4
		id_2_label={0:"conflict", 1:"negative", 2:"neutral", 3:"positive"}

		train_dataset = RLDSDataset(restaurant_file, tokenizer, 80, id_2_label, "train")
		valid_dataset = RLDSDataset(restaurant_file, tokenizer, 80, id_2_label, "valid")
		test_dataset = RLDSDataset(restaurant_file, tokenizer, 80, id_2_label, "test")

		dataset = {"train" : train_dataset,
				"validation" : valid_dataset,
				"test" : test_dataset}

	elif data_name == "laptop":
		data_path = "./Dataset/RLDS"
		l_train_v2 = "Laptop_Train_v2.csv"
		laptop_file = join(data_path, l_train_v2)

		num_of_labels = 4
		id_2_label={0:"conflict", 1:"negative", 2:"neutral", 3:"positive"}

		train_dataset = RLDSDataset(laptop_file, tokenizer, 90, id_2_label, "train")
		valid_dataset = RLDSDataset(laptop_file, tokenizer, 90, id_2_label, "valid")
		test_dataset = RLDSDataset(laptop_file, tokenizer, 90, id_2_label, "test")

		dataset = {"train" : train_dataset,
				"validation" : valid_dataset,
				"test" : test_dataset}

	elif data_name == "device":
		root_dir = "./Dataset/DLRS/"

		num_of_labels = 2
		id_2_label={0:"NEG", 1:"POS"}

		train_dataset = DLRSDataset(root_dir, "device", tokenizer, 400, id_2_label, mode="train")
		valid_dataset = DLRSDataset(root_dir, "device", tokenizer, 400, id_2_label, mode="valid")
		test_dataset = DLRSDataset(root_dir, "device", tokenizer, 400, id_2_label, mode="test")

		dataset = {"train" : train_dataset,
				"validation" : valid_dataset,
				"test" : test_dataset}

	elif data_name == "service":
		root_dir = "./Dataset/DLRS/"

		num_of_labels = 3
		id_2_label={0:"NEG", 1:"NEU", 2:"POS"}

		train_dataset = DLRSDataset(root_dir, "service", tokenizer, 350, id_2_label, mode="train")
		valid_dataset = DLRSDataset(root_dir, "service", tokenizer, 350, id_2_label, mode="valid")
		test_dataset = DLRSDataset(root_dir, "service", tokenizer, 350, id_2_label, mode="test")

		dataset = {"train" : train_dataset,
				"validation" : valid_dataset,
				"test" : test_dataset}

	elif data_name == "rest":   # restaurant in different split
		root_dir = "./Dataset/DLRS/"

		num_of_labels = 3
		id_2_label={0:"NEG", 1:"NEU", 2:"POS"}

		train_dataset = DLRSDataset(root_dir, "rest", tokenizer, 350, id_2_label, mode="train")
		valid_dataset = DLRSDataset(root_dir, "rest", tokenizer, 350, id_2_label, mode="valid")
		test_dataset = DLRSDataset(root_dir, "rest", tokenizer, 350, id_2_label, mode="test")

		dataset = {"train" : train_dataset,
				"validation" : valid_dataset,
				"test" : test_dataset}

	elif data_name == "new_laptop":
		root_dir = "./Dataset/DLRS/"

		num_of_labels = 3
		id_2_label={0:"NEG", 1:"NEU", 2:"POS"}

		train_dataset = DLRSDataset(root_dir, "laptop", tokenizer, 430, id_2_label, mode="train")
		valid_dataset = DLRSDataset(root_dir, "laptop", tokenizer, 430, id_2_label, mode="valid")
		test_dataset = DLRSDataset(root_dir, "laptop", tokenizer, 430, id_2_label, mode="test")

		dataset = {"train" : train_dataset,
				"validation" : valid_dataset,
				"test" : test_dataset}
	elif data_name in SUPERGLUE:
		dataset = SuperGlueDataset(tokenizer, args)
		num_of_labels = dataset.num_labels
		if data_name == "copa":
			id_2_label=None
		else:
			id_2_label = dataset.id2label

	elif data_name == "cb":
		train_dataset = load_dataset('super_glue', 'cb', split='train', cache_dir="./Dataset/super_glue/cb/")
		valid_dataset = load_dataset('super_glue', 'cb', split='validation', cache_dir="./Dataset/super_glue/cb/")
		# test_dataset = load_dataset('super_glue', 'cb', split='test', cache_dir="./Dataset/super_glue/cb/")
		test_dataset = load_dataset('super_glue', 'cb', split='validation', cache_dir="./Dataset/super_glue/cb/")
		
		train_dataset = train_dataset.rename_column("label", "labels")
		valid_dataset = valid_dataset.rename_column("label", "labels")
		test_dataset  = test_dataset.rename_column("label", "labels")

		num_of_labels = 3
		id_2_label={0:"0", 1:"1", 2:"2"}

		# Encode the input data
		train_dataset = train_dataset.map(cb_encode_batch, batched=True, remove_columns=train_dataset.column_names)
		valid_dataset = valid_dataset.map(cb_encode_batch, batched=True, remove_columns=valid_dataset.column_names)
		test_dataset  = test_dataset.map(cb_encode_batch, batched=True, remove_columns=test_dataset.column_names)

		# Transform to pytorch tensors and only output the required columns
		train_dataset.set_format(type="torch", columns=["input_ids",  "labels"])
		valid_dataset.set_format(type="torch", columns=["input_ids",  "labels"])
		test_dataset.set_format(type="torch", columns=["input_ids",  "labels"])

		dataset = {"train" : train_dataset,
				"validation" : valid_dataset,
				"test" : test_dataset}
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
		# per_device_eval_batch_size=args.batchsize,
		per_device_eval_batch_size=1,
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

	if args.dataset in SUPERGLUE:
		train_dataset = dataset.train_dataset
		valid_dataset = dataset.eval_dataset
		test_dataset = dataset.predict_dataset

		train_dataset = train_dataset.rename_column("label", "labels")
		valid_dataset = valid_dataset.rename_column("label", "labels")
		test_dataset = test_dataset.rename_column("label", "labels")
		print("====================================")
		print(train_dataset)
		print(valid_dataset)
		print(test_dataset)
		compute_metrics = dataset.compute_metrics
		data_collator = dataset.data_collator
	else:
		print(dataset)
		train_dataset = dataset["train"]
		valid_dataset = dataset["validation"]
		test_dataset = dataset["test"]
		data_collator = default_data_collator
		compute_metrics = compute_accuracy


	### Trainer
	trainer = AdapterTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=valid_dataset,
		tokenizer=tokenizer,
		data_collator=data_collator,
		compute_metrics=compute_metrics,
		callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)]
	)

	
	trainer.train()
	valid_metric = trainer.evaluate()

	print(valid_metric)
	print(trainer.predict(test_dataset).metrics)
	model.save_adapter(args.save_adapter_path, args.dataset)

	