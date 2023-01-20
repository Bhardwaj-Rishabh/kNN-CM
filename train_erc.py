
import torch
import numpy as np
from transformers import set_seed, TrainingArguments, AdapterTrainer, EvalPrediction
from transformers import RobertaTokenizer, RobertaConfig, RobertaModelWithHeads

from os.path import join

from transformers import EarlyStoppingCallback, default_data_collator
import logging
logging.disable(logging.WARNING)

from sklearn.metrics import precision_recall_fscore_support
import pickle as pk
from torch.utils.data import Dataset

set_seed(1234)

import random
random.seed(4)

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


class ERCDataset(Dataset):
	def __init__(self, args, mode, tokenizer):
		self.args = args
		data_name = args.dataset
		self.tokenizer = tokenizer
		dataset = pk.load(open(f"./datasets/{data_name}/{data_name}.pkl", "rb"))

		if mode == "train":
			texts = dataset["train_texts"]
			labels= dataset["train_labels"]
		elif mode == "valid":
			texts = dataset["val_texts"]
			labels= dataset["val_labels"]
		else:
			texts = dataset["test_texts"]
			labels= dataset["test_labels"]

		self.classes = [0,1,2,3,4,5]
		self.num_of_labels = len(self.classes)
		# breakpoint()

		self.texts = texts
		self.labels = labels

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, idx):
		text = self.texts[idx]
		label = self.labels[idx]
		return self.erc_encode_batch(text, label)

	def erc_encode_batch(self, text, label):
		outputs = self.tokenizer.encode(text,
							max_length=self.args.max_seq_length,
							truncation=True,
							padding="max_length")

		return {"input_ids": torch.tensor(outputs),
				"labels": torch.tensor(label)}



def get_dataset(tokenizer, args):

	train_dataset = ERCDataset(args,"train", tokenizer)
	valid_dataset = ERCDataset(args,"valid", tokenizer)
	test_dataset = ERCDataset(args,"test", tokenizer)
	num_of_labels = train_dataset.num_of_labels
	id_2_label = {class_:str(class_) for class_ in train_dataset.classes}
	dataset = {"train" : train_dataset,
			"validation" : valid_dataset,
			"test" : test_dataset}
	
	
	return dataset, num_of_labels, id_2_label

def compute_macro_f1(p: EvalPrediction):
	preds = np.argmax(p.predictions, axis=1)
	p_weighted, r_weighted, f_weighted, support_weighted = precision_recall_fscore_support(p.label_ids, preds, average='macro')
	return {"macro_f1":f_weighted}

def compute_accuracy(p: EvalPrediction):
	preds = np.argmax(p.predictions, axis=1)
	return {"accuracy": (preds == p.label_ids).mean()}

if __name__ == "__main__":

	### Args
	args = get_args()

	log_path = join(args.output_path, "log")
	training_args = TrainingArguments(
		learning_rate=args.lr,
		num_train_epochs=args.epochs,
		per_device_train_batch_size=args.batchsize,
		per_device_eval_batch_size=1,
		logging_steps=100, #25
		output_dir=args.output_path,
		overwrite_output_dir=True,
		load_best_model_at_end=True,
		metric_for_best_model=args.metric4train,
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
	dataset, num_of_labels, id_2_label = get_dataset(tokenizer, args)

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
	# breakpoint()

	#changed
	# Add a new adapter
	model.add_adapter(args.dataset)
	
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
		# callbacks = [TensorBoardCallback]
	)

	trainer.train()
	valid_metric = trainer.evaluate()

	print(valid_metric)
	print(trainer.predict(test_dataset).metrics)
	model.save_adapter(args.save_adapter_path, args.dataset)