from transformers import set_seed, RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, RobertaModelWithHeads
import numpy as np
from os.path import join
import argparse
from datasets import load_dataset, load_metric
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from transformers import EarlyStoppingCallback, default_data_collator
from sklearn.metrics import f1_score, classification_report
from collections import defaultdict

set_seed(1234)

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
	
	args = parser.parse_args()

	return args

def record_encode_batch(examples):
	encoded = defaultdict(list)
	for idx, passage, query, entities, answers in zip(
		examples["idx"][:2], examples["passage"][:2], examples["query"][:2], examples["entities"][:2], examples["answers"][:2]
	):
		for entity in entities:
			label = 1 if entity in answers else 0
			query_filled = query.replace("@placeholder", entity)
			example_encoded = tokenizer(
				passage,
				query_filled,
				max_length=512,
				truncation='only_first',
				padding="max_length",
				return_overflowing_tokens=True,
			)
			# if "overflowing_tokens" in example_encoded and len(example_encoded["overflowing_tokens"]) > 0:
			#	 logger.info("Cropping {0} tokens of input.".format(len(example_encoded["overflowing_tokens"])))
			encoded["idx"].append(idx)
			encoded["passage"].append(passage)
			encoded["query"].append(query_filled)
			encoded["entities"].append(entity)
			encoded["answers"].append(answers)
			encoded["input_ids"].append(example_encoded["input_ids"])
			encoded["labels"].append(label)
			if "token_type_ids" in example_encoded:
				encoded["token_type_ids"].append(example_encoded["token_type_ids"])
			if "attention_mask" in example_encoded:
				encoded["attention_mask"].append(example_encoded["attention_mask"])
	return encoded

_COPA_DICT = {
		"cause": "What was the cause of this?",
		"effect": "What happened as a result?",
	}

def copa_encode_batch_(examples):
	
	contexts = [p + " " + _COPA_DICT[q] for p, q in zip(examples["premise"], examples["question"])]
	sentences_a = [ctx + " " + choice for ctx, choice in zip(contexts, examples["choice1"])]
	sentences_b = [ctx + " " + choice for ctx, choice in zip(contexts, examples["choice2"])]
	encoded = tokenizer(
		sentences_a,
		sentences_b,
		max_length=512,
		truncation=True,
		padding="max_length",
		
	)
	encoded.update({"labels": examples["labels"]})
	return encoded

def copa_encode_batch(examples):
	examples["text_a"] = []
	for premise, question in zip(examples["premise"], examples["question"]):
		# joiner = "because" if question == "cause" else "so"
		joiner = _COPA_DICT[question]
		text_a = f"{premise} {joiner}"					
		examples["text_a"].append(text_a)

	result1 = tokenizer(examples["text_a"], examples["choice1"], padding="max_length", max_length=512, truncation=True) 
	result2 = tokenizer(examples["text_a"], examples["choice2"], padding="max_length", max_length=512, truncation=True)
	result = {}  
	for key in ["input_ids", "attention_mask", "token_type_ids"]:
		if key in result1 and key in result2:
			result[key] = []
			for value1, value2 in zip(result1[key], result2[key]):
				result[key].append([value1, value2])

	return result

def multirc_encode_batch(examples):
	contexts = [
		paragraph + " " + question for paragraph, question in zip(examples["paragraph"], examples["question"])
	]
	encoded = tokenizer(
		contexts,
		examples["answer"],
		max_length=512,
		# truncation=True,
		truncation='only_first',
		padding="max_length",
		return_overflowing_tokens=True
	)
	encoded.update({"labels": examples["labels"]})
	return encoded


def get_dataset(data_name, tokenizer, args):
	if data_name == "record":
		cache_path = f"./Dataset/super_glue/{data_name}"
		raw_datasets = load_dataset("super_glue", data_name, cache_dir=cache_path)
		# raw_datasets = raw_datasets.rename_column("label", "labels")

		num_of_labels = 2
		id_2_label={0:"0", 1:"1"}

		raw_datasets = raw_datasets.map(record_encode_batch, batched=True, remove_columns=raw_datasets["train"].column_names)
		raw_datasets.set_format(type="torch", columns=["input_ids",  "labels"])

		dataset = {"train" : raw_datasets["train"],
				"validation" : raw_datasets["validation"],
				"test" : raw_datasets["test"]}
	elif data_name == "copa":
		cache_path = f"./Dataset/super_glue/{data_name}"
		raw_datasets = load_dataset("super_glue", data_name, cache_dir=cache_path)
		raw_datasets = raw_datasets.rename_column("label", "labels")

		num_of_labels = 1
		id_2_label=None

		raw_datasets = raw_datasets.map(copa_encode_batch, batched=True)#, remove_columns=raw_datasets["train"].column_names)
		raw_datasets.set_format(type="torch", columns=["input_ids",  "labels"])

		dataset = {"train" : raw_datasets["train"],
				"validation" : raw_datasets["validation"],
				"test" : raw_datasets["test"]}

	elif data_name == "multirc":
		cache_path = f"./Dataset/super_glue/{data_name}"
		raw_datasets = load_dataset("super_glue", data_name, cache_dir=cache_path)
		raw_datasets = raw_datasets.rename_column("label", "labels")

		num_of_labels = 2
		id_2_label={0:"0", 1:"1"}

		raw_datasets = raw_datasets.map(multirc_encode_batch, batched=True)#, remove_columns=raw_datasets["train"].column_names)
		raw_datasets.set_format(type="torch", columns=["input_ids",  "labels"])

		dataset = {"train" : raw_datasets["train"],
				"validation" : raw_datasets["validation"],
				"test" : raw_datasets["test"]}
	else:
		raise NotImplementedError

	return dataset, num_of_labels, id_2_label


args = get_args()
dataset_name = args.dataset.lower()
#### Dataset
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
dataset, num_of_labels, id_2_label = get_dataset(args.dataset, tokenizer, args)

### Config
if dataset_name == "copa":
	config = RobertaConfig.from_pretrained("roberta-base", num_choices=2)
else:
	config = RobertaConfig.from_pretrained("roberta-base", num_labels=num_of_labels)
	
### Model
model = RobertaModelWithHeads.from_pretrained("roberta-base",config=config)

adapter_name = model.load_adapter(f"AdapterHub/roberta-base-pf-{dataset_name}", source="hf")
model.active_adapters = adapter_name

log_path = join(args.output_path, "log")
training_args = TrainingArguments(
	learning_rate=args.lr,
	num_train_epochs=args.epochs,
	per_device_train_batch_size=args.batchsize,
	per_device_eval_batch_size=args.batchsize,
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

train_dataset = dataset["train"]
valid_dataset = dataset["validation"]
test_dataset = dataset["test"]

def compute_f1(p: EvalPrediction):
	if isinstance(p.predictions, tuple):
		p.predictions = p.predictions[0]
	preds = np.argmax(p.predictions, axis=1)
	print(classification_report( p.label_ids, preds, digits=4))
	return {"f1": f1_score(p.label_ids, preds)}

def multirc_metirc(p: EvalPrediction):
	if isinstance(p.predictions, tuple):
		p.predictions = p.predictions[0]
	predictions = np.argmax(p.predictions, axis=1)
	predictions = [{"idx": idx, "prediction": pred} for idx, pred in zip(valid_dataset["idx"], predictions)]
	metric = load_metric("super_glue", "multirc")
	return metric.compute(predictions=predictions, references=p.label_ids)

def record_metric(p: EvalPrediction):
	if isinstance(p.predictions, tuple):
		p.predictions = p.predictions[0]
	predictions = p.predictions
	max_preds = {}  # group predictions by question id
	for idx, entity, pred, answers in zip(
		valid_dataset["idx"], valid_dataset["entities"], predictions, valid_dataset["answers"]
	):
		idx_string = f"{idx['passage']}-{idx['query']}"
		if idx_string not in max_preds or pred[1] > max_preds[idx_string]["logit"]:
			max_preds[idx_string] = {"idx": idx, "logit": pred[1], "entity": entity, "answers": answers}
	predictions = [{"idx": val["idx"], "prediction_text": val["entity"]} for _, val in max_preds.items()]
	references = [{"idx": val["idx"], "answers": val["answers"]} for _, val in max_preds.items()]
	metric = load_metric("super_glue", "record")
	return metric.compute(predictions=predictions, references=references)

def copa_metric(p: EvalPrediction):
	if isinstance(p.predictions, tuple):
		p.predictions = p.predictions[0]
	predictions = np.argmax(p.predictions, axis=1)
	metric = load_metric("super_glue", "copa")
	return metric.compute(predictions=predictions, references=p.label_ids)

if dataset_name == "record":
	compute_metric = record_metric
elif dataset_name == "multirc":
	compute_metric = multirc_metirc
elif dataset_name == "copa":
	compute_metric = copa_metric
else:
	raise NotImplementedError

### Trainer
trainer = AdapterTrainer(
	model=model,
	args=training_args,
	train_dataset=train_dataset,
	eval_dataset=valid_dataset,
	tokenizer=tokenizer,
	data_collator=default_data_collator,
	compute_metrics=compute_metric,
	callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)]
	# callbacks = [TensorBoardCallback]
)

print("------->>>>>>>>>>>>>>.", trainer.predict(valid_dataset).metrics)

