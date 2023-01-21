#[script to construct datastores]

import os
import sys
import time
import itertools
import numpy as np

import torch
from tqdm import tqdm
from transformers import set_seed, RobertaTokenizer, RobertaModelWithHeads, EvalPrediction

import faiss
import psutil

from os.path import join
from preprocess import RLDSDataset

from datasets import Dataset
from datasets import load_dataset, load_metric
from datasets import load_from_disk
from datasets import concatenate_datasets

from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter

from scipy.special import softmax

import argparse
from scipy.special import rel_entr

import pickle as pk

set_seed(1314)
import random
random.seed(4)

def get_args():
	parser = argparse.ArgumentParser(description='')

	parser.add_argument('--epoch', type=int)
	parser.add_argument('--split', type=str)
	parser.add_argument('--in_dataset', type=str)
	parser.add_argument('--out_dataset', type=str)
	parser.add_argument('--model_id', type=str)
	parser.add_argument('--layer_id', type=int)
	parser.add_argument('--dstore_path', type=str)
	parser.add_argument('--create_dstore', type=bool, default=False)
	parser.add_argument('--use_adapter', action='store_true')
	parser.add_argument('--adapter_path', type=str, default="./saved_adapters/")
	parser.add_argument('--num_proc', type=int, default=-1)
	parser.add_argument('--num_labels', type=int, default=6)
	parser.add_argument('--lambdas', type=list, default=[0.5]) #1e-3, 1e-2, 0.05, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
	parser.add_argument('--topn', type=list, default=[32]) #1, 2, 4, 8, 16, 32, 64, 128, 256, 512
	parser.add_argument('--kl_thresholds', type=list, default=[0.3]) #0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
	parser.add_argument('--pad_to_max_length', type=bool, default=True)
	parser.add_argument('--max_seq_length', type=int, default=128)
	parser.add_argument('--use_2dstores', type=bool, default=False)

	args = parser.parse_args()

	return args

#register hook to extract input to final ffl after layer norm
activation = {}
def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook
	
def faiss_read(args):
	# Read the index from disk
	# if args.create_dstore:
	# 	index_name = "./indexs/" + args.in_dataset + "2" + args.out_dataset + "_index_layer" + str(args.layer_id)
	# else:
	# 	index_name = "./indexs/" + args.in_dataset + "_index_layer" + str(args.layer_id)

	index_name = "./indexs/" + args.in_dataset + "_index_layer" + str(args.layer_id)
	# index_name = "./indexs/" + args.in_dataset + "2" + args.out_dataset + "_index_noadapter_layer" + str(args.layer_id)
	# index_name = "./indexs/" + args.in_dataset + "2" + args.out_dataset + "_index_layer" + str(args.layer_id)

	index = faiss.read_index(index_name)

	return index

def faiss_read_2dstores(args, hidden_dim):

	out_task_dstore = args.dstore_path + args.in_dataset + "2" + args.out_dataset + "_dstore_" + str(args.layer_id)
	in_task_dstore = args.dstore_path + args.in_dataset + "_dstore_" + str(args.layer_id)

	in_out_2dstores = args.dstore_path + args.in_dataset + "2" + args.out_dataset + "_in_out_2dstores_" + str(args.layer_id)

	in_num_samples = 162865
	out_num_samples = 250

	dstore_keys_in = np.memmap(in_task_dstore + '_key.npy', dtype=np.float16, mode='r', shape=(in_num_samples, hidden_dim))
	dstore_keys_out = np.memmap(out_task_dstore + '_key.npy', dtype=np.float16, mode='r', shape=(out_num_samples, hidden_dim))

	dstore_embeds = np.concatenate((dstore_keys_in, dstore_keys_out), axis=0)
	import copy
	dstore_keys = copy.deepcopy(dstore_embeds)

	###### Build Faiss #####
	# Define the size of index vectors
	vector_dimension = hidden_dim

	# Create the Flat L2 index
	index = faiss.IndexFlatL2(vector_dimension)

	# Add vectors to index. Embeddings can be Numpy arrays or torch tensors
	index.add(dstore_keys)

	index_name = "./indexs/" + args.in_dataset + "2" + args.out_dataset + "_2dstores_" + "_index_layer" + str(args.layer_id)

	faiss.write_index(index, index_name)

	index = faiss.read_index(index_name)
	return index


def create_datastore(args, num_samples, train_datasets, model):
	dataset_name = args.out_dataset
	'''Allocate datastore memory'''	
	dstore_filename = args.dstore_path + args.in_dataset + "2" + dataset_name + "_dstore_" + str(args.layer_id)
	# if args.use_adapter:
	# 	dstore_filename = args.dstore_path + args.in_dataset + "2" + dataset_name + "_dstore_" + str(args.layer_id)
	# else:
	# 	dstore_filename = args.dstore_path + args.in_dataset + "2" + dataset_name + "_dstore_noadapter_" + str(args.layer_id)
	# dstore_filename = args.dstore_path + args.in_dataset + "2" + dataset_name + "_dstore_noadapter_" + str(args.layer_id)

	if dataset_name == "copa":
		hidden_dim = model.config.hidden_size * 2
	else:
		hidden_dim = model.config.hidden_size
	
	#create datastore
	dstore_keys = np.memmap(dstore_filename + '_key.npy', dtype=np.float16, mode='w+', shape=(num_samples, hidden_dim))
	dstore_vals = np.memmap(dstore_filename + '_val.npy', dtype=np.int32, mode='w+', shape=(num_samples, 1))

	'''Construct datastore'''
	# method1
	model.roberta.encoder.layer[args.layer_id].attention.output.LayerNorm.register_forward_hook(get_activation(f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'))
	
	#store elements into datastore
	for i in tqdm(range(0, num_samples)):
		if dataset_name == "copa":
			shape1 = 2
		else:
			shape1 = 1
		input_ids = train_datasets[i]['input_ids'].clone().detach().view(shape1,-1).to(model.device)
		target_ids = train_datasets[i]['labels'].clone().detach().view(-1).to(model.device)

		with torch.no_grad():
			if "attention_mask" in train_datasets[i].keys():
				attention_mask = train_datasets[i]["attention_mask"].to(model.device)
			else:
				attention_mask = None
			#obtain context context representations
			# outputs = model(input_ids, labels=target_ids)
			outputs = model(input_ids, attention_mask, labels=target_ids)

			## write context vectors
			### method1
			if dataset_name == "copa":
				dstore_keys[i, :] = activation[f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'][:,0,:].reshape(1,-1).squeeze(0).cpu().numpy().astype(np.float16)
			else:
				dstore_keys[i, :] = activation[f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'].squeeze(0)[0].cpu().numpy().astype(np.float16)
			#write value/next word token  #changed
			dstore_vals[i, ] = target_ids[0].cpu().numpy().astype(np.intc)
	
	print(f"\n\n\t\tDatastore construction done! and saved to {dstore_filename}")
	print(f"\n\n\t\tStarting to build faiss index...")

	# breakpoint()

	
	###### Build Faiss #####
	# Define the size of index vectors
	vector_dimension = hidden_dim

	# Create the Flat L2 index
	index = faiss.IndexFlatL2(vector_dimension)

	# Add vectors to index. Embeddings can be Numpy arrays or torch tensors
	index.add(dstore_keys)

	# index_name = "./indexs/" + args.in_dataset + "2" + dataset_name + "_index_noadapter_layer" + str(args.layer_id)
	index_name = "./indexs/" + args.in_dataset + "2" + dataset_name + "_index_layer" + str(args.layer_id)

	faiss.write_index(index, index_name)

def get_test_acc_knnonly(args, test_datasets, index, num_labels, model):
	y_true = []
	neighbours_matrix=[]
	neighbour_labels_matrix=[]
	neighbours_distance_matrix=[]

	dstore_filename = args.dstore_path + args.in_dataset + "2" + args.out_dataset + "_dstore_noadapter_" + str(args.layer_id)
	train_labels = np.memmap(dstore_filename + '_val.npy', dtype=np.int32, mode='r')

	# method1
	model.roberta.encoder.layer[args.layer_id].attention.output.LayerNorm.register_forward_hook(get_activation(f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'))

	knn_need_count=0
	for i, data in enumerate(test_datasets):
		input_ids = test_datasets[i]['input_ids'].clone().detach().view(1,-1).to(model.device)
		target_ids = test_datasets[i]['labels'].clone().detach().view(-1).to(model.device)

		with torch.no_grad():
			#obtain context context representations
			if "attention_mask" in data.keys():
				attention_mask = data["attention_mask"].to(model.device)
			else:
				attention_mask = None
			if args.out_dataset == "copa":
				input_ids = input_ids.view(-1, args.max_seq_length)
			lm_output = model(input_ids, attention_mask, labels=target_ids)

			embedding = activation[f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'].squeeze(0)[0].cpu().numpy().astype(np.float16)

			embedding = np.expand_dims(embedding, axis=0)
			distances, neighbours = index.search(embedding, k=512)   # shape: (1,4)

			neighbour_labels = []
			for idx in neighbours[0]:
				idx = idx.item()
				lab = train_labels[idx]
				neighbour_labels.append(lab)

			y_true.append(target_ids.squeeze(0).cpu().tolist())
			neighbours_matrix.append(neighbours[0].tolist())
			neighbour_labels_matrix.append(neighbour_labels)
			neighbours_distance_matrix.append(distances[0])

	results = []
	exp_idx = 0

	for topk in args.topn:
		for lambda_ in args.lambdas:
			for kl_threshold in args.kl_thresholds:

				tmp = {"idx":exp_idx, "topk":topk, "lambda":lambda_, "kl_threshold":kl_threshold}

				print("*"*60)
				print("\n top_k_neighbours: ", topk, " lambda: ", lambda_, " kl_threshold: ", kl_threshold,  " layer_id:", args.layer_id, "\n")
				print("*"*60)

				knn_need_count=0
				knn_logits = []
				y_knn_preds = []

				for j in range(len(test_datasets)):
					neighbour_labels = neighbour_labels_matrix[j][:topk]
					counter = Counter(neighbour_labels)
					num_all = len(neighbour_labels)
					knn_logit = [counter[i]*1.0/num_all for i in range(num_labels)]
					knn_logit = np.array(knn_logit)
					knn_logits.append(knn_logit)

					y_knn_pred = np.argmax(knn_logit, axis=-1).tolist()  
					y_knn_preds.append(y_knn_pred)

				labels = [i for i in range(num_labels)]
				print("\n#num of under kl_threshold: ", knn_need_count, "\n")
				print("================ KNN only======================")
				print(classification_report(y_true, y_knn_preds, labels=labels, digits=4))
				knn_logits = np.stack(knn_logits,0)

				if args.out_dataset == "cb":
					metrics = cb_metrics(predictions=knn_logits, references=y_true)
				else:
					raise NotImplementedError
				print(metrics)

def get_test_acc(args, test_datasets, index, num_labels, model):
	y_true = []
	y_pred = []
	lm_logits = []
	neighbours_matrix=[]
	neighbour_labels_matrix=[]
	neighbours_distance_matrix=[]

	# dstore_filename = args.dstore_path + args.in_dataset + "2" + args.out_dataset + "_dstore_noadapter_" + str(args.layer_id)
	# dstore_filename = args.dstore_path + args.in_dataset + "2" + args.out_dataset + "_dstore_" + str(args.layer_id)
	dstore_filename = args.dstore_path + args.in_dataset + "_dstore_" + str(args.layer_id)
	train_labels = np.memmap(dstore_filename + '_val.npy', dtype=np.int32, mode='r')

	# if args.create_dstore:
	# 	dstore_filename = args.dstore_path + args.in_dataset + "2" + args.out_dataset + "_dstore_" + str(args.layer_id)
	# else:
	# 	dstore_filename = args.dstore_path + args.in_dataset + "_dstore_" + str(args.layer_id)

	# if args.use_2dstores:
	# 	out_task_dstore = args.dstore_path + args.in_dataset + "2" + args.out_dataset + "_dstore_" + str(args.layer_id)
	# 	in_task_dstore = args.dstore_path + args.in_dataset + "_dstore_" + str(args.layer_id)

	# 	train_labels_in = np.memmap(in_task_dstore + '_val.npy', dtype=np.int32, mode='r')
	# 	train_labels_out = np.memmap(out_task_dstore + '_val.npy', dtype=np.int32, mode='r')

	# 	# breakpoint()

	# 	train_labels = np.concatenate((train_labels_in, train_labels_out), axis=0)
	# else:
	# 	train_labels = np.memmap(dstore_filename + '_val.npy', dtype=np.int32, mode='r')

	# breakpoint()

	# method1
	model.roberta.encoder.layer[args.layer_id].attention.output.LayerNorm.register_forward_hook(get_activation(f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'))

	knn_need_count=0
	for i, data in enumerate(test_datasets):
		if args.out_dataset == "copa":
			shape1 = 2
		else:
			shape1 = 1
		input_ids = test_datasets[i]['input_ids'].clone().detach().view(shape1,-1).to(model.device)
		target_ids = test_datasets[i]['labels'].clone().detach().view(-1).to(model.device)

		with torch.no_grad():
			#obtain context context representations
			if "attention_mask" in data.keys():
				attention_mask = data["attention_mask"].to(model.device)
			else:
				attention_mask = None
			if args.out_dataset == "copa":
				input_ids = input_ids.view(-1, args.max_seq_length)
			# lm_output = model(input_ids, labels=target_ids)
			# breakpoint()
			lm_output = model(input_ids, attention_mask, labels=target_ids)
			logits = torch.softmax(lm_output.logits, dim=-1)
			lm_logit = logits.detach().cpu().numpy()[0]
			# breakpoint()
			
			if args.in_dataset == "anli" and args.out_dataset in ["axb", "axg"]:
				lm_logit = np.array([lm_logit[0], lm_logit[1] + lm_logit[2]]) # axb
				# lm_logit = np.array([lm_logit[0] + lm_logit[1], lm_logit[2]])
				# lm_logit = np.array([max(lm_logit[0],lm_logit[1]), lm_logit[2]])  # axg
				# lm_logit = np.array([0, 1])
				logits  = torch.tensor(lm_logit).unsqueeze(0)
			else:
				lm_logit = lm_logit
			# breakpoint()
			lm_logits.append(lm_logit)
			
			# Search the top-k nearest neighbors of all the vectors in embedding
			# method1
			if args.out_dataset == "copa":
				embedding = activation[f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'][:,0,:].reshape(1,-1).squeeze(0).cpu().numpy().astype(np.float16)
			else:
				embedding = activation[f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'].squeeze(0)[0].cpu().numpy().astype(np.float16)
			
			embedding = np.expand_dims(embedding, axis=0)
			distances, neighbours = index.search(embedding, k=512)   # shape: (1,4)

			neighbour_labels = []
			for idx in neighbours[0]:
				idx = idx.item()
				lab = train_labels[idx]
				neighbour_labels.append(lab)

			y_lm_pred = torch.argmax(logits, dim=-1).cpu().tolist()

			y_true.append(target_ids.squeeze(0).cpu().tolist())
			y_pred.append(y_lm_pred)
			neighbours_matrix.append(neighbours[0].tolist())
			neighbour_labels_matrix.append(neighbour_labels)
			neighbours_distance_matrix.append(distances[0])

	if args.in_dataset == "anli" and args.out_dataset in ["axb", "axg"]:
		uniform_dis = [1.0/2 for i in range(2)]
	else:
		uniform_dis = [1.0/num_labels for i in range(num_labels)]

	results = []
	exp_idx = 0

	for topk in args.topn:
		for lambda_ in args.lambdas:
			for kl_threshold in args.kl_thresholds:

				tmp = {"idx":exp_idx, "topk":topk, "lambda":lambda_, "kl_threshold":kl_threshold}

				print("*"*60)
				print("\n top_k_neighbours: ", topk, " lambda: ", lambda_, " kl_threshold: ", kl_threshold,  " layer_id:", args.layer_id, "\n")
				print("*"*60)

				knn_need_count=0
				knn_logits = []
				knn_lm_logits = []
				y_knn_lm_pred = []
				y_knn_preds = []

				for j in range(len(lm_logits)):
					neighbour_labels = neighbour_labels_matrix[j][:topk]
					counter = Counter(neighbour_labels)
					num_all = len(neighbour_labels)
					knn_logit = [counter[i]*1.0/num_all for i in range(num_labels)]
					knn_logit = np.array(knn_logit)

					if args.in_dataset == "anli" and args.out_dataset in ["axb", "axg"]:
						# knn_logit = np.array([knn_logit[0], knn_logit[1]+knn_logit[2]]) 
						# knn_logit = np.array([knn_logit[0] + knn_logit[1], knn_logit[2]]) 
						knn_logit = np.array([max(knn_logit[0], knn_logit[1]), knn_logit[2]]) 
					else:
						knn_logit = knn_logit

					knn_logits.append(knn_logit)

					kl_divergence = sum(rel_entr(lm_logits[j], uniform_dis))
						
					if kl_divergence < kl_threshold :
						knn_need_count += 1
						knn_lm_logit = lambda_ * lm_logits[j] + (1-lambda_) * knn_logit
					else:
						knn_lm_logit = lm_logits[j]

					# breakpoint()
					y_knn_pred = np.argmax(knn_logit, axis=-1).tolist()  
					y_knn_preds.append(y_knn_pred)

					######## format {out = lambda*(p_model) + (1-lamba) *(p_fassi)}
					# knn_lm_logit = lambda_ * lm_logit + (1-lambda_) * knn_logit
					knn_lm_logits.append(knn_lm_logit)

					y_knnlm_pred = np.argmax(knn_lm_logit, axis=-1).tolist()  
					y_knn_lm_pred.append(y_knnlm_pred)

				# labels = [i for i in range(num_labels)]
				labels = [1,2,3]
				print("\n#num of under kl_threshold: ", knn_need_count, "\n")
				print("================ LM only======================")
				print(classification_report(y_true, y_pred, labels=labels, digits=4))
				# print(classification_report(y_true, y_pred, digits=4))

				lm_logits = np.stack(lm_logits,0)
				if args.out_dataset == "axg":
					metrics = axg_metrics(predictions=lm_logits, references=y_true)
				elif args.out_dataset == "axb":
					metrics = axb_metrics(predictions=lm_logits, references=y_true)
				elif args.out_dataset == "cb":
					metrics = cb_metrics(predictions=lm_logits, references=y_true)
				elif args.out_dataset in ["restaurant", "laptop", "anli"]:
					metrics = compute_accuracy(predictions=lm_logits, references=y_true)
				else:
					raise NotImplementedError
				print(metrics)
				print("================ KNN only======================")
				print(classification_report(y_true, y_knn_preds, labels=labels, digits=4))

				print("================ KNN-LM ======================")
				print(classification_report(y_true, y_knn_lm_pred, labels=labels, digits=4))
				# print(classification_report(y_true, y_knn_lm_pred, digits=4))
				
				knn_lm_logits = np.stack(knn_lm_logits,0)
				if args.out_dataset == "axg":
					metrics = axg_metrics(predictions=knn_lm_logits, references=y_true)
					tmp["metrics"] = metrics["accuracy"]
				elif args.out_dataset == "axb":
					metrics = axb_metrics(predictions=knn_lm_logits, references=y_true)
					tmp["metrics"] = metrics["matthews_correlation"]
				elif args.out_dataset == "cb":
					metrics = cb_metrics(predictions=knn_lm_logits, references=y_true)
					tmp["metrics"] = metrics["accuracy"]
				elif args.out_dataset in ["restaurant", "laptop", "anli"]:
					metrics = compute_accuracy(predictions=knn_lm_logits, references=y_true)
					tmp["metrics"] = metrics["accuracy"]
				else:
					raise NotImplementedError
				print(metrics)

				results.append(tmp)
				exp_idx = exp_idx + 1

	sorted_result = sorted(results, key=lambda k: k["metrics"], reverse=True)
	print(sorted_result[:10])

''' Tokenize the dataset'''
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
def axg_encode_batch(examples):
	encoded = tokenizer(
		examples["premise"],
		examples["hypothesis"],
		max_length=512,
		truncation=True,
		padding="max_length",
	)
	encoded.update({"labels": examples["labels"]})
	return encoded

def axb_encode_batch(examples):
	encoded = tokenizer(
		examples["sentence1"],
		examples["sentence2"],
		max_length=512,
		truncation=True,
		padding="max_length",
	)
	encoded.update({"labels": examples["labels"]})
	return encoded

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
	
	return {"idx":batch["uid"],
			"input_ids": outputs,
			"labels": batch["labels"]}

def axg_metrics(predictions, references):
	if isinstance(predictions, tuple):
		predictions = predictions[0]
	predictions = np.argmax(predictions, axis=1)
	metric = load_metric("super_glue", "axg")
	return metric.compute(predictions=predictions, references=references)

def axb_metrics(predictions, references):
	if isinstance(predictions, tuple):
		predictions = predictions[0]
	predictions = np.argmax(predictions, axis=1)
	metric = load_metric("super_glue", "axb")
	return metric.compute(predictions=predictions, references=references)

def cb_metrics(predictions, references):
	if isinstance(predictions, tuple):
		predictions = predictions[0]
	predictions = np.argmax(predictions, axis=1)
	metric = load_metric("super_glue", "cb")
	return metric.compute(predictions=predictions, references=references)

def compute_accuracy(predictions, references):
	if isinstance(predictions, tuple):
		predictions = predictions[0]
	predictions = np.argmax(predictions, axis=1)
	metric = load_metric("accuracy")
	return metric.compute(predictions=predictions, references=references)


if __name__ == "__main__":

	args = get_args()
	in_dataset  = args.in_dataset.lower()
	out_dataset = args.out_dataset.lower()

	if not os.path.exists(args.dstore_path):
		os.makedirs(args.dstore_path)

	'''Load pre-trained language model'''
	model = RobertaModelWithHeads.from_pretrained(args.model_id, output_hidden_states=True, )
	tokenizer = RobertaTokenizer.from_pretrained(args.model_id)

	if args.use_adapter:
		print("\n\n\tLoading adapter...\n")
		adapter_name = model.load_adapter(f"{args.adapter_path}")
		model.active_adapters = adapter_name
		# model.load_adapter(f"{args.adapter_path}")
		# model.set_active_adapters(None)

	print(model)

	if args.num_proc==-1:
		args.num_proc = psutil.cpu_count()

	'''Load dataset'''
	if out_dataset == "axg":
		cache_path = f"./Dataset/super_glue/{out_dataset}"
		raw_datasets = load_dataset("super_glue", out_dataset, cache_dir=cache_path)
		raw_datasets = raw_datasets.rename_column("label", "labels")
		print(raw_datasets)

		raw_datasets = raw_datasets.map(axg_encode_batch, batched=True)#, remove_columns=raw_datasets["train"].column_names)
		raw_datasets.set_format(type="torch", columns=["input_ids",  "labels"])

		train_datasets=None
		test_datasets = raw_datasets["test"]
	elif out_dataset == "axb":
		cache_path = f"./Dataset/super_glue/{out_dataset}"
		raw_datasets = load_dataset("super_glue", out_dataset, cache_dir=cache_path)
		raw_datasets = raw_datasets.rename_column("label", "labels")
		print(raw_datasets)

		raw_datasets = raw_datasets.map(axb_encode_batch, batched=True)
		raw_datasets.set_format(type="torch", columns=["input_ids",  "labels"])

		train_datasets=None
		test_datasets = raw_datasets["test"]
	elif out_dataset == "cb":
		train_dataset = load_dataset('super_glue', 'cb', split='train', cache_dir="./Dataset/super_glue/cb/")
		valid_dataset = load_dataset('super_glue', 'cb', split='validation', cache_dir="./Dataset/super_glue/cb/")
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

		train_datasets = train_dataset
		test_datasets = valid_dataset
	elif out_dataset == "restaurant":
		data_path = "./Dataset/RLDS"
		r_train_v2 = "Restaurants_Train_v2.csv"
		restaurant_file = join(data_path, r_train_v2)

		num_of_labels = 4
		id_2_label={0:"conflict", 1:"negative", 2:"neutral", 3:"positive"}

		train_dataset = RLDSDataset(restaurant_file, tokenizer, 80, id_2_label, "train")
		valid_dataset = RLDSDataset(restaurant_file, tokenizer, 80, id_2_label, "valid")
		test_dataset = RLDSDataset(restaurant_file, tokenizer, 80, id_2_label, "test")

		train_datasets = train_dataset
		valid_datasets = valid_dataset
		test_datasets = test_dataset
	elif out_dataset == "laptop":
		laptop_file = "./Dataset/RLDS/Laptop_Train_v2.csv"
		id_2_label={0:"conflict", 1:"negative", 2:"neutral", 3:"positive"}

		train_datasets = RLDSDataset(laptop_file, tokenizer, 90, id_2_label, "train")
		valid_datasets = RLDSDataset(laptop_file, tokenizer, 90, id_2_label, "valid")
		test_datasets = RLDSDataset(laptop_file, tokenizer, 90, id_2_label, "test")

	elif out_dataset == "anli":
		# num_of_labels = 3
		# id_2_label={0:"0", 1:"1", 2:"2"}

		dataset = load_dataset("anli", cache_dir="./Dataset/anli/")
		dataset = dataset.rename_column("label", "labels")
		
		dataset = dataset.map(anli_encode_batch, batched=True, remove_columns=dataset["train_r2"].column_names)
		dataset.set_format(type="torch", columns=["input_ids",  "labels"])
		
		train_datasets = torch.utils.data.ConcatDataset([dataset["train_r1"], dataset["train_r2"], dataset["train_r3"]])
		valid_datasets = torch.utils.data.ConcatDataset([dataset["dev_r1"], dataset["dev_r2"], dataset["dev_r3"]])
		test_datasets = torch.utils.data.ConcatDataset([dataset["test_r1"], dataset["test_r2"], dataset["test_r3"]])
	else:
		raise NotImplementedError

	model = model.cuda()

	if args.create_dstore == True:
		'''Create Datastore and Build Faiss'''
		create_datastore(args, num_samples=len(train_datasets), train_datasets=train_datasets, model=model)

	# '''Faiss Read Index'''
	# if args.use_2dstores == True:
	# 	index = faiss_read_2dstores(args, model.config.hidden_size)
	# else:
	# 	index = faiss_read(args)

	index = faiss_read(args)

	'''Test Dataset Acc'''
	get_test_acc(args, test_datasets, index, args.num_labels, model)
	







