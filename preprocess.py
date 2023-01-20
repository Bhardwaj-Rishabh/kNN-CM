from os.path import join
import csv
from torch.utils.data import Dataset #, DataLoader
import string

from transformers import set_seed
set_seed(1234)
import random
random.seed(4)



class RLDSDataset(Dataset):
	def __init__(self, csv_file, tokenizer, max_length, id_2_label, mode):  # id, Sentence, Aspect Term, polarity, from, to'
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.id_2_label = id_2_label
		self.label_2_id = {v:k for k,v in id_2_label.items()}
		samples = []
		with open(csv_file, newline='') as csvfile:
			reader = csv.reader(csvfile)
			line_count = 0
			for i, row in enumerate(reader):
				if row[3] == "conflict":
					# breakpoint()
					continue
				if line_count == 0:
					line_count += 1
				else:
					line_count += 1
					samples.append({ "id":row[0], "sentence":row[1], "aspect_term":row[2], "polarity":row[3], "from":row[4], "to":row[5]})

		num_samples = len(samples)
		train_len = int(num_samples * 0.4)
		valid_len = int(num_samples * 0.3)
		test_len = num_samples - train_len - valid_len

		# Shuffle
		random.shuffle(samples)

		if mode == "train":
			self.samples = samples[:train_len]
		elif mode == "valid":
			self.samples = samples[train_len:train_len+valid_len]
		elif mode == "test":
			self.samples = samples[-test_len:]
		else:
			raise NotImplementedError

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		return self.rlds_encode_batch(sample)

	def rlds_encode_batch(self, sample):
		"""Encodes a batch of input data using the model tokenizer."""

		from_ = int(sample["from"])
		to_ = int(sample["to"])
		new_sentence = sample["sentence"][:from_] + "<" + sample["sentence"][from_:to_] + ">" + sample["sentence"][to_:]

		outputs = self.tokenizer(new_sentence, max_length=self.max_length, truncation=True, padding="max_length")

		
		return {"input_ids": torch.tensor(outputs["input_ids"]),
				"attention_mask": torch.tensor(outputs["attention_mask"]),
				"labels": torch.tensor(self.label_2_id[sample["polarity"]])}
		

def ot2bio_ote(ote_tag_sequence):
	"""
	ot2bio function for ote tag sequence
	:param ote_tag_sequence:
	:return:
	"""
	new_ote_sequence = []
	n_tag = len(ote_tag_sequence)
	prev_ote_tag = '$$$'
	for i in range(n_tag):
		cur_ote_tag = ote_tag_sequence[i]
		assert cur_ote_tag == 'O' or cur_ote_tag == 'T'
		if cur_ote_tag == 'O':
			new_ote_sequence.append(cur_ote_tag)
		else:
			# cur_ote_tag is T
			if prev_ote_tag == 'T':
				new_ote_sequence.append('I')
			else:
				# cur tag is at the beginning of the opinion target
				new_ote_sequence.append('B')
		prev_ote_tag = cur_ote_tag
	return new_ote_sequence


def ot2bio_ts(ts_tag_sequence):
	"""
	ot2bio function for ts tag sequence
	:param ts_tag_sequence:
	:return:
	"""
	new_ts_sequence = []
	n_tag = len(ts_tag_sequence)
	prev_pos = '$$$'
	for i in range(n_tag):
		cur_ts_tag = ts_tag_sequence[i]
		if cur_ts_tag == 'O':
			new_ts_sequence.append('O')
			cur_pos = 'O'
		else:
			# current tag is subjective tag, i.e., cur_pos is T
			# print(cur_ts_tag)
			cur_pos, cur_sentiment = cur_ts_tag.split('-')
			if cur_pos == prev_pos:
				# prev_pos is T
				new_ts_sequence.append('I-%s' % cur_sentiment)
			else:
				# prev_pos is O
				new_ts_sequence.append('B-%s' % cur_sentiment)
		prev_pos = cur_pos
	return new_ts_sequence

def ot2bieos_ote(ote_tag_sequence):
	"""
	ot2bieos function for ote task
	:param ote_tag_sequence:
	:return:
	"""
	n_tags = len(ote_tag_sequence)
	new_ote_sequence = []
	prev_ote_tag = '$$$'
	for i in range(n_tags):
		cur_ote_tag = ote_tag_sequence[i]
		if cur_ote_tag == 'O':
			new_ote_sequence.append('O')
		else:
			# cur_ote_tag is T
			if prev_ote_tag != cur_ote_tag:
				# prev_ote_tag is O, new_cur_tag can only be B or S
				if i == n_tags - 1:
					new_ote_sequence.append('S')
				elif ote_tag_sequence[i + 1] == cur_ote_tag:
					new_ote_sequence.append('B')
				elif ote_tag_sequence[i + 1] != cur_ote_tag:
					new_ote_sequence.append('S')
				else:
					raise Exception("Invalid ner tag value: %s" % cur_ote_tag)
			else:
				# prev_tag is T, new_cur_tag can only be I or E
				if i == n_tags - 1:
					new_ote_sequence.append('E')
				elif ote_tag_sequence[i + 1] == cur_ote_tag:
					# next_tag is T
					new_ote_sequence.append('I')
				elif ote_tag_sequence[i + 1] != cur_ote_tag:
					# next_tag is O
					new_ote_sequence.append('E')
				else:
					raise Exception("Invalid ner tag value: %s" % cur_ote_tag)
		prev_ote_tag = cur_ote_tag
	return new_ote_sequence

def ot2bieos_ts(ts_tag_sequence):
	"""
	ot2bieos function for ts task
	:param ts_tag_sequence: tag sequence for targeted sentiment
	:return:
	"""
	n_tags = len(ts_tag_sequence)
	new_ts_sequence = []
	prev_pos = '$$$'
	for i in range(n_tags):
		cur_ts_tag = ts_tag_sequence[i]
		if cur_ts_tag == 'O':
			new_ts_sequence.append('O')
			cur_pos = 'O'
		else:
			cur_pos, cur_sentiment = cur_ts_tag.split('-')
			# cur_pos is T
			if cur_pos != prev_pos:
				# prev_pos is O and new_cur_pos can only be B or S
				if i == n_tags - 1:
					new_ts_sequence.append('S-%s' % cur_sentiment)
				else:
					next_ts_tag = ts_tag_sequence[i + 1]
					if next_ts_tag == 'O':
						new_ts_sequence.append('S-%s' % cur_sentiment)
					else:
						new_ts_sequence.append('B-%s' % cur_sentiment)
			else:
				# prev_pos is T and new_cur_pos can only be I or E
				if i == n_tags - 1:
					new_ts_sequence.append('E-%s' % cur_sentiment)
				else:
					next_ts_tag = ts_tag_sequence[i + 1]
					if next_ts_tag == 'O':
						new_ts_sequence.append('E-%s' % cur_sentiment)
					else:
						new_ts_sequence.append('I-%s' % cur_sentiment)
		prev_pos = cur_pos
	return new_ts_sequence


def ot2bieos(ote_tag_sequence, ts_tag_sequence):
	"""
	perform ot-->bieos for both ote tag and ts tag sequence
	:param ote_tag_sequence: input tag sequence of opinion target extraction
	:param ts_tag_sequence: input tag sequence of targeted sentiment
	:return:
	"""
	# new tag sequences of opinion target extraction and targeted sentiment
	new_ote_sequence = ot2bieos_ote(ote_tag_sequence=ote_tag_sequence)
	new_ts_sequence = ot2bieos_ts(ts_tag_sequence=ts_tag_sequence)
	assert len(ote_tag_sequence) == len(new_ote_sequence)
	assert len(ts_tag_sequence) == len(new_ts_sequence)
	return new_ote_sequence, new_ts_sequence


def ot2bio(ote_tag_sequence, ts_tag_sequence):
	"""
	perform ot--->bio for both ote tag sequence and ts tag sequence
	:param ote_tag_sequence: input tag sequence of opinion target extraction
	:param ts_tag_sequence: input tag sequence of targeted sentiment
	:return:
	"""
	new_ote_sequence = ot2bio_ote(ote_tag_sequence=ote_tag_sequence)
	new_ts_sequence = ot2bio_ts(ts_tag_sequence=ts_tag_sequence)
	assert len(new_ts_sequence) == len(ts_tag_sequence)
	assert len(new_ote_sequence) == len(ote_tag_sequence)
	return new_ote_sequence, new_ts_sequence

class DLRSDataset(Dataset):
	def __init__(self, root_dir, dataset, tokenizer, max_length, id_2_label, mode):  # id, Sentence, Aspect Term, polarity, from, to'
		if mode == "test":
			path = join(root_dir, f"{dataset}_test.txt")
		else:
			path = join(root_dir, f"{dataset}_train.txt")
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.id_2_label = id_2_label
		self.label_2_id = {v:k for k,v in id_2_label.items()}

		samples = self.read_data(path)

		num_samples = len(samples)
		train_len = int(num_samples * 0.8)
		valid_len = num_samples - train_len

		# Shuffle
		# random.shuffle(samples)

		if mode == "train":
			samples = samples[:train_len]
		elif mode == "valid":
			samples = samples[train_len:]
		elif mode == "test":
			samples = samples
		else:
			raise NotImplementedError

		samples, ote_tag_vocab, ts_tag_vocab = self.set_labels(dataset=samples, tagging_schema="BIEOS")

		self.samples = []

		for idx, sample in enumerate(samples):
			aspect_terms_ids = []
			polarities = []
			skip_flag = False

			for i,ote_label in enumerate(sample["ote_tags"]):
				if ote_label == "O":
					pass
				elif ote_label == "B":
					tmp = []
					tmp.append(i)
				elif ote_label == "I":
					tmp.append(i)
				elif ote_label == "E":
					tmp.append(i)
					aspect_terms_ids.append(tmp)
				elif ote_label == "S":
					aspect_terms_ids.append([i])
				else:
					raise "Wrong opinion target extraction vocab!"

			for at in aspect_terms_ids:
				polarity = list(set([sample["ts_raw_tags"][j].split("-")[-1] for j in at]))
				
				try :
					assert len(polarity) == 1
				except:
					skip_flag = True

				polarities.append(polarity)

			if skip_flag:
				continue
			else:
				assert len(polarities) == len(aspect_terms_ids)

				if len(aspect_terms_ids) > 1:
					for m, aspect_term_id in enumerate(aspect_terms_ids):
						aspect_term = [sample["words"][z] for z in aspect_term_id]
						polarity = polarities[m][0]
						from_ = aspect_term_id[0]
						if len(aspect_term_id) > 1:
							to = aspect_term_id[-1]
						else:
							to = from_ + 1
						self.samples.append({ "sentence": sample["words"], "aspect_term":" ".join(aspect_term), "polarity":polarity, "from":from_, "to":to})
				else:
					aspect_term = [sample["words"][z] for z in aspect_terms_ids[0]]
					polarity = polarities[0][0]
					from_ = aspect_terms_ids[0][0]
					if len(aspect_terms_ids[0]) > 1:
						to = aspect_terms_ids[0][-1] + 1
					else:
						to = from_ + 1
					self.samples.append({ "sentence": sample["words"], "aspect_term":" ".join(aspect_term), "polarity":polarity, "from":from_, "to":to})

	def read_data(self, path):
		"""
		read data from the specified path
		:param path: path of dataset
		:return:
		"""
		dataset = []
		with open(path) as fp:
			for line in fp:
				record = {}
				# print(line.strip())
				# print(line.strip().split('####'))
				sent, tag_string = line.strip().split('####')

				record['sentence'] = sent
				word_tag_pairs = tag_string.split(' ')
				# tag sequence for targeted sentiment
				ts_tags = []
				# tag sequence for opinion target extraction
				ote_tags = []
				# word sequence
				words = []
				for item in word_tag_pairs:
					# valid label is: O, T-POS, T-NEG, T-NEU
					eles = item.split('=')
					if len(eles) == 2:
						word, tag = eles
					elif len(eles) > 2:
						tag = eles[-1]
						word = (len(eles) - 2) * "="
					if word not in string.punctuation:
						# lowercase the words
						words.append(word.lower())
					else:
						# replace punctuations with a special token
						words.append('PUNCT')
					if tag == 'O':
						ote_tags.append('O')
						ts_tags.append('O')
					elif tag == 'T-POS':
						ote_tags.append('T')
						ts_tags.append('T-POS')
					elif tag == 'T-NEG':
						ote_tags.append('T')
						ts_tags.append('T-NEG')
					elif tag == 'T-NEU':
						ote_tags.append('T')
						ts_tags.append('T-NEU')
					else:
						raise Exception('Invalid tag %s!!!' % tag)
				flag = False
				for ote_tag in ote_tags:
					if ote_tag != 'O':
						flag = True
						break
				# flag = True
				if flag:
					record['words'] = list(words)
					record['ote_raw_tags'] = list(ote_tags)
					record['ts_raw_tags'] = list(ts_tags)
					dataset.append(record)
		print("Obtain %s records from %s" % (len(dataset), path))
		return dataset

	def set_labels(self, dataset, tagging_schema='BIO'):
		"""
		set ote_label and ts_label for the dataset
		:param dataset: dataset without ote_label and ts_label fields
		:param tagging_schema: tagging schema of ote_tag and ts_tag
		:return:
		"""
		if tagging_schema == 'OT':
			ote_tag_vocab = {'O': 0, 'T': 1}
			ts_tag_vocab = {'O': 0, 'T-POS': 1, 'T-NEG': 2, 'T-NEU': 3}
		elif tagging_schema == 'BIO':
			ote_tag_vocab = {'O': 0, 'B': 1, 'I': 2}
			ts_tag_vocab = {'O': 0, 'B-POS': 1, 'I-POS': 2, 'B-NEG': 3, 'I-NEG': 4,
							'B-NEU': 5, 'I-NEU': 6}
		elif tagging_schema == 'BIEOS':
			ote_tag_vocab = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
			ts_tag_vocab = {'O': 0, 'B-POS': 1, 'I-POS': 2, 'E-POS': 3, 'S-POS': 4,
							'B-NEG': 5, 'I-NEG': 6, 'E-NEG': 7, 'S-NEG': 8,
							'B-NEU': 9, 'I-NEU': 10, 'E-NEU': 11, 'S-NEU': 12}
		else:
			raise Exception("Invalid tagging schema %s" % tagging_schema)
		n_records = len(dataset)
		for i in range(n_records):
			ote_tags = dataset[i]['ote_raw_tags']
			ts_tags = dataset[i]['ts_raw_tags']
			if tagging_schema == 'OT':
				pass
			elif tagging_schema == 'BIO':
				ote_tags, ts_tags = ot2bio(ote_tag_sequence=ote_tags, ts_tag_sequence=ts_tags)
			elif tagging_schema == 'BIEOS':
				ote_tags, ts_tags = ot2bieos(ote_tag_sequence=ote_tags, ts_tag_sequence=ts_tags)
			else:
				raise Exception("Invalid tagging schema %s" % tagging_schema)
			ote_labels = [ote_tag_vocab[t] for t in ote_tags]
			ts_labels = [ts_tag_vocab[t] for t in ts_tags]
			dataset[i]['ote_tags'] = list(ote_tags)
			dataset[i]['ts_tags'] = list(ts_tags)
			dataset[i]['ote_labels'] = list(ote_labels)
			dataset[i]['ts_labels'] = list(ts_labels)
		return dataset, ote_tag_vocab, ts_tag_vocab
	
	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		return self.dlrs_encode_batch(sample)

	def dlrs_encode_batch(self, sample):
		"""Encodes a batch of input data using the model tokenizer."""

		from_ = int(sample["from"])
		to_ = int(sample["to"])
		
		new_sentence = " ".join(sample["sentence"][:from_]) + " < " + " ".join(sample["sentence"][from_:to_]) + " > " + " ".join(sample["sentence"][to_:])

		outputs = self.tokenizer(new_sentence, max_length=self.max_length, truncation=True, padding="max_length")

		return {"input_ids": outputs["input_ids"],
				"attention_mask": outputs["attention_mask"],
				"labels": self.label_2_id[sample["polarity"]]}


from datasets.load import load_dataset, load_metric
from transformers import (
	AutoTokenizer,
	DataCollatorWithPadding,
	EvalPrediction,
	default_data_collator,
)
import numpy as np
import logging
from collections import defaultdict

task_to_keys = {
	"boolq": ("question", "passage"),
	"cb": ("premise", "hypothesis"),
	"rte": ("premise", "hypothesis"),
	"wic": ("processed_sentence1", None),
	"wsc": ("span2_word_text", "span1_text"),
	"copa": (None, None),
	"record": (None, None),
	# "multirc": ("paragraph", "question_answer")
	"multirc": ("paragraph_question", "answer")
}

logger = logging.getLogger(__name__)

"""
	Code from https://github.com/THUDM/P-tuning-v2/blob/main/tasks/superglue/dataset.py
"""
class SuperGlueDataset():
	def __init__(self, tokenizer: AutoTokenizer, data_args) -> None:
		super().__init__()
		cache_path = f"/data/yingting/Dataset/super_glue/{data_args.dataset}"
		raw_datasets = load_dataset("super_glue", data_args.dataset, cache_dir=cache_path)
		self.tokenizer = tokenizer
		self.data_args = data_args

		self.multiple_choice = data_args.dataset in ["copa"]

		if data_args.dataset == "record":
			self.num_labels = 2
			self.label_list = ["0", "1"]
		elif not self.multiple_choice:
			self.label_list = raw_datasets["train"].features["label"].names
			self.num_labels = len(self.label_list)
		else:
			self.num_labels = 1

		# Preprocessing the raw_datasets
		self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset]

		# Padding strategy
		if data_args.pad_to_max_length:
			self.padding = "max_length"
		else:
			# We will pad later, dynamically at batch creation, to the max sequence length in each batch
			self.padding = False

		if not self.multiple_choice:
			self.label2id = {l: i for i, l in enumerate(self.label_list)}
			self.id2label = {id: label for label, id in self.label2id.items()}
			print(f"{self.label2id}")
			print(f"{self.id2label}")

		if data_args.max_seq_length > tokenizer.model_max_length:
			logger.warning(
				f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
				f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
			)
		self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

		if data_args.dataset == "record":
			raw_datasets = raw_datasets.map(
				self.record_preprocess_function,
				batched=True,
				# load_from_cache_file=not data_args.overwrite_cache,
				remove_columns=raw_datasets["train"].column_names,
				desc="Running tokenizer on dataset",
			)
		else:
			raw_datasets = raw_datasets.map(
				self.preprocess_function,
				batched=True,
				# load_from_cache_file=not data_args.overwrite_cache,
				desc="Running tokenizer on dataset",
			)

		self.train_dataset = raw_datasets["train"]
		self.eval_dataset = raw_datasets["validation"]
		# self.predict_dataset = raw_datasets["test"]
		self.predict_dataset = raw_datasets["validation"]

		
		
		self.metric = load_metric("super_glue", data_args.dataset)

		if data_args.pad_to_max_length:
			self.data_collator = default_data_collator
		elif data_args.fp16:
			self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

		self.test_key = "accuracy" if data_args.dataset not in ["record", "multirc"] else "f1"

	def preprocess_function(self, examples):
		examples["paragraph"] = examples["paragraph"][:5]
		examples["question"] = examples["question"][:5]
		examples["answer"] = examples["answer"][:5]
		examples["label"] = examples["label"][:5]

		# WSC
		if self.data_args.dataset == "wsc":
			examples["span2_word_text"] = []
			for text, span2_index, span2_word in zip(examples["text"], examples["span2_index"], examples["span2_text"]):
				examples["span2_word_text"].append(span2_word + ": " + text)
				# if self.data_args.template_id == 0:
				# 	examples["span2_word_text"].append(span2_word + ": " + text)
				# elif self.data_args.template_id == 1:
				# 	words_a = text.split()
				# 	words_a[span2_index] = "*" + words_a[span2_index] + "*"
				# 	examples["span2_word_text"].append(' '.join(words_a))

		# WiC
		if self.data_args.dataset == "wic":
			examples["processed_sentence1"] = []
			# if self.data_args.template_id == 1:
			# 	self.sentence2_key = "processed_sentence2"
			# 	examples["processed_sentence2"] = []
			for sentence1, sentence2, word, start1, end1, start2, end2 in zip(examples["sentence1"], examples["sentence2"], examples["word"], examples["start1"], examples["end1"], examples["start2"], examples["end2"]):
				examples["processed_sentence1"].append(f"{sentence1} {sentence2} Does {word} have the same meaning in both sentences?")
				# if self.data_args.template_id == 0: #ROBERTA
				# 	examples["processed_sentence1"].append(f"{sentence1} {sentence2} Does {word} have the same meaning in both sentences?")
				# elif self.data_args.template_id == 1: #BERT
				# 	examples["processed_sentence1"].append(word + ": " + sentence1)
				# 	examples["processed_sentence2"].append(word + ": " + sentence2)

		# MultiRC
		if self.data_args.dataset == "multirc":
			# print("========================")
			# print(len(examples["idx"]))
			# print(len(examples["paragraph"]))
			# print(len(examples["question"]))
			# print(len(examples["answer"]))
			# exit()
			del examples["idx"]
			
			# examples["question_answer"] = []
			# for question, asnwer in zip(examples["question"], examples["answer"]):
			# 	examples["question_answer"].append(f"{question}"+ " <combination> "+ f"{asnwer}")

			examples["paragraph_question"] = []
			for paragraph, question in zip(examples["paragraph"], examples["question"]):
				examples["paragraph_question"].append(f"{paragraph} {question}")
			

		# COPA
		if self.data_args.dataset == "copa":
			examples["text_a"] = []
			for premise, question in zip(examples["premise"], examples["question"]):
				joiner = "because" if question == "cause" else "so"
				text_a = f"{premise} {joiner}"					
				examples["text_a"].append(text_a)

			result1 = self.tokenizer(examples["text_a"], examples["choice1"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
			result2 = self.tokenizer(examples["text_a"], examples["choice2"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
			result = {}  
			for key in ["input_ids", "attention_mask", "token_type_ids"]:
				if key in result1 and key in result2:
					result[key] = []
					for value1, value2 in zip(result1[key], result2[key]):
						result[key].append([value1, value2])
			return result

		# if self.data_args.dataset == "multirc":
		# 	for question_asnwer in examples["question_answer"]:
				

		args = (
			(examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
		)
		result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

		return result

	def compute_metrics(self, p: EvalPrediction):
		preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
		preds = np.argmax(preds, axis=1)

		if self.data_args.dataset == "record":
			return self.reocrd_compute_metrics(p)

		if self.data_args.dataset == "multirc":
			print("==================preds==========================")
			print(preds)
			print("==================p.label_ids==========================")
			print(p.label_ids)
			from sklearn.metrics import f1_score, precision_recall_fscore_support
			return {"f1": f1_score(preds, p.label_ids)}
			
			# p_weighted, r_weighted, f_weighted, support_weighted = precision_recall_fscore_support(p.label_ids, preds, average='macro')
			# return {"macro_f1":f_weighted}

		if self.data_args.dataset == "wsc":
			from sklearn.metrics import precision_recall_fscore_support
			result = self.metric.compute(predictions=preds, references=p.label_ids)
			p_weighted, r_weighted, f_weighted, support_weighted = precision_recall_fscore_support(p.label_ids, preds, average='macro')
			result["f1"] = f_weighted
			return result
			

		if self.data_args.dataset is not None:
			result = self.metric.compute(predictions=preds, references=p.label_ids)
			if len(result) > 1:
				result["combined_score"] = np.mean(list(result.values())).item()
			return result
		elif self.is_regression:
			return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
		else:
			return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

	def reocrd_compute_metrics(self, p: EvalPrediction):
		from superglue_utils import f1_score, exact_match_score, metric_max_over_ground_truths
		probs = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
		examples = self.eval_dataset
		qid2pred = defaultdict(list)
		qid2ans = {}
		for prob, example in zip(probs, examples):
			qid = example['question_id']
			qid2pred[qid].append((prob[1], example['entity']))
			if qid not in qid2ans:
				qid2ans[qid] = example['answers']
		n_correct, n_total = 0, 0
		f1, em = 0, 0
		for qid in qid2pred:
			preds = sorted(qid2pred[qid], reverse=True)
			entity = preds[0][1]
			n_total += 1
			n_correct += (entity in qid2ans[qid])
			f1 += metric_max_over_ground_truths(f1_score, entity, qid2ans[qid])
			em += metric_max_over_ground_truths(exact_match_score, entity, qid2ans[qid])
		acc = n_correct / n_total
		f1 = f1 / n_total
		em = em / n_total
		return {'f1': f1, 'exact_match': em}

	def record_preprocess_function(self, examples, split="train"):
		results = {
			# "index": list(),
			"question_id": list(),
			"input_ids": list(),
			"attention_mask": list(),
			# "token_type_ids": list(),
			"label": list(),
			"entity": list(),
			"answers": list()
		}
		for idx, passage in enumerate(examples["passage"]):
			query, entities, answers =  examples["query"][idx], examples["entities"][idx], examples["answers"][idx]
			index = examples["idx"][idx]
			passage = passage.replace("@highlight\n", "- ")
			
			for ent_idx, ent in enumerate(entities):
				question = query.replace("@placeholder", ent)
				result = self.tokenizer(passage, question, padding=self.padding, max_length=self.max_seq_length, truncation=True)
				label = 1 if ent in answers else 0

				results["input_ids"].append(result["input_ids"])
				results["attention_mask"].append(result["attention_mask"])
				# if "token_type_ids" in result: results["token_type_ids"].append(result["token_type_ids"])
				results["label"].append(label)
				# results["index"].append(index)
				results["question_id"].append(index["query"])
				results["entity"].append(ent)
				results["answers"].append(answers)

		return results	

import torch
from torch.utils import data
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_dataset, load_metric
from transformers import (
	AutoTokenizer,
	DataCollatorWithPadding,
	EvalPrediction,
	default_data_collator,
	DataCollatorForLanguageModeling
)
import random
import numpy as np
import logging

from dataclasses import dataclass
from transformers.data.data_collator import DataCollatorMixin
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

logger = logging.getLogger(__name__)

@dataclass
class DataCollatorForMultipleChoice(DataCollatorMixin):
	tokenizer: PreTrainedTokenizerBase
	padding: Union[bool, str, PaddingStrategy] = True
	max_length: Optional[int] = None
	pad_to_multiple_of: Optional[int] = None
	label_pad_token_id: int = -100
	return_tensors: str = "pt"

	def torch_call(self, features):
		label_name = "label" if "label" in features[0].keys() else "labels"
		labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
		batch = self.tokenizer.pad(
			features,
			padding=self.padding,
			max_length=self.max_length,
			pad_to_multiple_of=self.pad_to_multiple_of,
			# Conversion to tensors will fail if we have labels as they are not of the same length yet.
			return_tensors="pt" if labels is None else None,
		)

		if labels is None:
			return batch

		sequence_length = torch.tensor(batch["input_ids"]).shape[1]
		padding_side = self.tokenizer.padding_side
		if padding_side == "right":
			batch[label_name] = [
				list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
			]
		else:
			batch[label_name] = [
				[self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
			]

		batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
		print(batch)
		input_list = [sample['input_ids'] for sample in batch]

		choice_nums = list(map(len, input_list))
		max_choice_num = max(choice_nums)

		def pad_choice_dim(data, choice_num):
			if len(data) < choice_num:
				data = np.concatenate([data] + [data[0:1]] * (choice_num - len(data)))
			return data

		for i, sample in enumerate(batch):
			for key, value in sample.items():
				if key != 'label':
					sample[key] = pad_choice_dim(value, max_choice_num)
				else:
					sample[key] = value
			# sample['loss_mask'] = np.array([1] * choice_nums[i] + [0] * (max_choice_num - choice_nums[i]),
			#								dtype=np.int64)

		return batch


class SuperGlueDatasetForRecord(SuperGlueDataset):
	def __init__(self, tokenizer: AutoTokenizer, data_args) -> None:
		cache_path = f"/data/yingting/Dataset/super_glue/{data_args.dataset}"
		raw_datasets = load_dataset("super_glue", data_args.dataset, cache_dir=cache_path)
		self.tokenizer = tokenizer
		self.data_args = data_args
		#labels
		self.multiple_choice = data_args.dataset in ["copa", "record"]

		if not self.multiple_choice:
			self.label_list = raw_datasets["train"].features["label"].names
			self.num_labels = len(self.label_list)
		else:
			self.num_labels = 1

		# Padding strategy
		if data_args.pad_to_max_length:
			self.padding = "max_length"
		else:
			# We will pad later, dynamically at batch creation, to the max sequence length in each batch
			self.padding = False

		# Some models have set the order of the labels to use, so let's make sure we do use it.
		self.label_to_id = None

		if self.label_to_id is not None:
			self.label2id = self.label_to_id
			self.id2label = {id: label for label, id in self.label2id.items()}
		elif not self.multiple_choice:
			self.label2id = {l: i for i, l in enumerate(self.label_list)}
			self.id2label = {id: label for label, id in self.label2id.items()}


		if data_args.max_seq_length > tokenizer.model_max_length:
			logger.warning(
				f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
				f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
			)
		self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

		
		self.train_dataset = raw_datasets["train"]
		self.train_dataset = self.train_dataset.map(
			self.prepare_train_dataset,
			batched=True,
			# load_from_cache_file=not data_args.overwrite_cache,
			remove_columns=raw_datasets["train"].column_names,
			desc="Running tokenizer on train dataset",
		)
		
		self.eval_dataset = raw_datasets["validation"]
		self.eval_dataset = self.eval_dataset.map(
			self.prepare_eval_dataset,
			batched=True,
			# load_from_cache_file=not data_args.overwrite_cache,
			remove_columns=raw_datasets["train"].column_names,
			desc="Running tokenizer on validation dataset",
		)

		self.predict_dataset = self.eval_dataset
			
		self.metric = load_metric("super_glue", data_args.dataset)

		self.data_collator = DataCollatorForMultipleChoice(tokenizer)
		# if data_args.pad_to_max_length:
		#	 self.data_collator = default_data_collator
		# elif training_args.fp16:
		#	 self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
	def preprocess_function(self, examples):
		results = {
			"input_ids": list(),
			"attention_mask": list(),
			"token_type_ids": list(),
			"label": list()
		}
		for passage, query, entities, answers in zip(examples["passage"], examples["query"], examples["entities"], examples["answers"]):
			passage = passage.replace("@highlight\n", "- ")

			input_ids = []
			attention_mask = []
			token_type_ids = []
			
			for _, ent in enumerate(entities):
				question = query.replace("@placeholder", ent)
				result = self.tokenizer(passage, question, padding=self.padding, max_length=self.max_seq_length, truncation=True)
				
				input_ids.append(result["input_ids"])
				attention_mask.append(result["attention_mask"])
				if "token_type_ids" in result: token_type_ids.append(result["token_type_ids"])
				label = 1 if ent in answers else 0
			
			result["label"].append()

		return results


	def prepare_train_dataset(self, examples, max_train_candidates_per_question=10):
		entity_shuffler = random.Random(44)
		results = {
			"input_ids": list(),
			"attention_mask": list(),
			"token_type_ids": list(),
			"label": list()
		}
		for passage, query, entities, answers in zip(examples["passage"], examples["query"], examples["entities"], examples["answers"]):
			passage = passage.replace("@highlight\n", "- ")
			
			for answer in answers:
				input_ids = []
				attention_mask = []
				token_type_ids = []
				candidates = [ent for ent in entities if ent not in answers]
				# if len(candidates) < max_train_candidates_per_question - 1:
				#	 continue
				if len(candidates) > max_train_candidates_per_question - 1:
					entity_shuffler.shuffle(candidates)
					candidates = candidates[:max_train_candidates_per_question - 1]
				candidates = [answer] + candidates

				for ent in candidates:
					question = query.replace("@placeholder", ent)
					result = self.tokenizer(passage, question, padding=self.padding, max_length=self.max_seq_length, truncation=True)
					input_ids.append(result["input_ids"])
					attention_mask.append(result["attention_mask"])
					if "token_type_ids" in result: token_type_ids.append(result["token_type_ids"])

				results["input_ids"].append(input_ids)
				results["attention_mask"].append(attention_mask)
				if len(token_type_ids) > 0: results["token_type_ids"].append(token_type_ids)
				results["label"].append(0)

		return results
			

	def prepare_eval_dataset(self, examples):

		results = {
			"input_ids": list(),
			"attention_mask": list(),
			"token_type_ids": list(),
			"label": list()
		}
		for passage, query, entities, answers in zip(examples["passage"], examples["query"], examples["entities"], examples["answers"]):
			passage = passage.replace("@highlight\n", "- ")
			for answer in answers:
				input_ids = []
				attention_mask = []
				token_type_ids = []

				for ent in entities:
					question = query.replace("@placeholder", ent)
					result = self.tokenizer(passage, question, padding=self.padding, max_length=self.max_seq_length, truncation=True)
					input_ids.append(result["input_ids"])
					attention_mask.append(result["attention_mask"])
					if "token_type_ids" in result: token_type_ids.append(result["token_type_ids"])

				results["input_ids"].append(input_ids)
				results["attention_mask"].append(attention_mask)
				if len(token_type_ids) > 0: results["token_type_ids"].append(token_type_ids)
				results["label"].append(0)

		return results



if __name__ == "__main__":
	
	from transformers import RobertaTokenizer
	from collections import Counter
	from train1 import get_args

	args = get_args()
	args.dataset = "record"

	tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

	# dataset = SuperGlueDataset(tokenizer, args)
	dataset = SuperGlueDatasetForRecord(tokenizer, args)

	

	


	'''
		DLRS
	'''
	# root_dir = "/data/yingting/Dataset/DLRS/"

	#Device
	# train = DLRSDataset(root_dir, "device", tokenizer, 400, id_2_label={0:"NEG", 1:"NEU", 2:"POS"}, mode="train")
	# valid = DLRSDataset(root_dir, "device", tokenizer, 400, id_2_label={0:"NEG", 1:"NEU", 2:"POS"}, mode="valid")
	# test = DLRSDataset(root_dir, "device", tokenizer, 400, id_2_label={0:"NEG", 1:"NEU", 2:"POS"}, mode="test")
	# print("======= Device Dataset ============")

	#Service
	# train = DLRSDataset(root_dir, "service", tokenizer, 350, id_2_label={0:"NEG", 1:"NEU", 2:"POS"}, mode="train")
	# valid = DLRSDataset(root_dir, "service", tokenizer, 350, id_2_label={0:"NEG", 1:"NEU", 2:"POS"}, mode="valid")
	# test  = DLRSDataset(root_dir, "service", tokenizer, 350, id_2_label={0:"NEG", 1:"NEU", 2:"POS"}, mode="test")
	# print("======= Service Dataset ============")

	#Restaurant
	# train = DLRSDataset(root_dir, "rest", tokenizer, 350, id_2_label={0:"NEG", 1:"NEU", 2:"POS"}, mode="train")
	# valid = DLRSDataset(root_dir, "rest", tokenizer, 350, id_2_label={0:"NEG", 1:"NEU", 2:"POS"}, mode="valid")
	# test = DLRSDataset(root_dir, "rest", tokenizer, 350, id_2_label={0:"NEG", 1:"NEU", 2:"POS"}, mode="test")
	# print("======= Restaurant Dataset ============")

	#Laptop
	# train = DLRSDataset(root_dir, "laptop", tokenizer, 430, id_2_label={0:"NEG", 1:"NEU", 2:"POS"}, mode="train")
	# valid = DLRSDataset(root_dir, "laptop", tokenizer, 430, id_2_label={0:"NEG", 1:"NEU", 2:"POS"}, mode="valid")
	# test = DLRSDataset(root_dir, "laptop", tokenizer, 430, id_2_label={0:"NEG", 1:"NEU", 2:"POS"}, mode="test")
	# print("======= Laptop Dataset ============")

	# labels = []

	# for sample in train:
	# 	labels.append(sample["labels"])

	# print("train: ", Counter(labels))

	# valid_labels = []
	
	# for sample in valid:
	# 	valid_labels.append(sample["labels"])

	# print("valid: ", Counter(valid_labels))

	# test_labels = []
	# for sample in test:
	# 	test_labels.append(sample["labels"])

	# print("test: ", Counter(test_labels))

	'''
		RLDS
	'''

	# data_path = "/data/yingting/Dataset/RLDS"

	### Restaurants
	# r_train_v2 = "Restaurants_Train_v2.csv"
	# restaurant_file = join(data_path, r_train_v2)
	# train_dataset = RLDSDataset(restaurant_file, tokenizer, "train")
	# valid_dataset = RLDSDataset(restaurant_file, "valid")
	# test_dataset = RLDSDataset(restaurant_file, "test")
	# print("\n----------------------\n")
	# print("len of train_dataset:", len(train_dataset))
	# print("len of valid_dataset:", len(valid_dataset))
	# print("len of test_dataset :", len(test_dataset))
	# print("\n----------- train_dataset -----------\n")
	# print(train_dataset[0])
	# print("\n----------- valid_dataset -----------\n")
	# print(valid_dataset[0])
	# print("\n----------- test_dataset -----------\n")
	# print(test_dataset[0])


	### Laptop
	# l_train_v2 = "Laptop_Train_v2.csv"
	# laptop_file = join(data_path, l_train_v2)
	# train_dataset = RLDSDataset(laptop_file, tokenizer, 300, "train")
	# valid_dataset = RLDSDataset(laptop_file, "valid")
	# test_dataset = RLDSDataset(laptop_file, "test")
	# print("\n----------------------\n")
	# print("len of train_dataset:", len(train_dataset))
	# print("len of valid_dataset:", len(valid_dataset))
	# print("len of test_dataset :", len(test_dataset))
	# print("\n----------- train_dataset -----------\n")
	# print(train_dataset[0])
	# print("\n----------- valid_dataset -----------\n")
	# print(valid_dataset[0])
	# print("\n----------- test_dataset -----------\n")
	# print(test_dataset[0])





