import os
import time
import numpy as np

import torch
from tqdm import tqdm
from transformers import set_seed, RobertaTokenizer, RobertaModelWithHeads

import faiss
import psutil

from datasets import load_dataset, load_metric

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter


import argparse
from scipy.special import rel_entr
from scipy.special import kl_div
from preprocess import RLDSDataset, DLRSDataset, SuperGlueDataset

import pickle as pk

# set_seed(1314)
set_seed(1234)
import random
random.seed(4)

SUPERGLUE = {"rte", "boolq", "wic", "wsc"}


def get_args():
	parser = argparse.ArgumentParser(description='')

	parser.add_argument('--epoch', type=int)
	parser.add_argument('--split', type=str)
	parser.add_argument('--dataset', type=str)
	parser.add_argument('--model_id', type=str)
	parser.add_argument('--layer_id', type=int)
	parser.add_argument('--dstore_path', type=str)
	parser.add_argument('--create_dstore', type=bool, default=False)
	parser.add_argument('--use_adapter', action='store_true')
	parser.add_argument('--adapter_path', type=str, default="./saved_adapters/")
	parser.add_argument('--num_proc', type=int, default=-1)
	parser.add_argument('--num_labels', type=int, default=6)
	parser.add_argument('--lambdas', type=list, default=[0.5]) #1e-3, 1e-2, 0.05, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
	parser.add_argument('--topn', type=list, default=[4]) #1, 2, 4, 8, 16, 32, 64, 128, 256, 512
	parser.add_argument('--kl_thresholds', type=list, default=[10]) #0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
	parser.add_argument('--pad_to_max_length', type=bool, default=True)
	parser.add_argument('--max_seq_length', type=int, default=128)

	args = parser.parse_args()

	return args


''' Tokenize the dataset'''
def tokenize(element):
	#preprocess
	outputs = tokenizer(
		element["text"],
		truncation=False
	)

	indices = [i for i,l in enumerate(outputs['input_ids']) if len(l)>0] #find sequences with non zero tokens

	outputs = {'input_ids':[outputs['input_ids'][i] for i in indices], 
				'attention_mask':[outputs['attention_mask'][i] for i in indices],
				'labels': [element["labels"][i] for i in indices]}
				# 'text' : [element["text"][i] for i in indices]}

	return outputs

''' Tokenize the anli'''
def multi_input_tokenize(batch):
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


#register hook to extract input to final ffl after layer norm
activation = {}
def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook

# def create_datastore(num_samples, hidden_dim, dstore_path, dataset_name, model, device):
from scipy.special import softmax
def create_datastore(args, num_samples, train_datasets, model):
	'''Allocate datastore memory'''	
	dstore_filename = args.dstore_path + args.dataset + "_dstore_" + str(args.layer_id)
	if args.dataset=="copa":
		hidden_dim = 1536
	else:
		hidden_dim = model.config.hidden_size

	#create datastore
	dstore_keys = np.memmap(dstore_filename + '_key.npy', dtype=np.float16, mode='w+', shape=(num_samples, hidden_dim))
	dstore_vals = np.memmap(dstore_filename + '_val.npy', dtype=np.int32, mode='w+', shape=(num_samples, 1))

	'''Construct datastore'''
	# method1
	model.roberta.encoder.layer[args.layer_id].attention.output.LayerNorm.register_forward_hook(get_activation(f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'))
	# method2
	# model.roberta.encoder.layer[args.layer_id].output.LayerNorm.register_forward_hook(get_activation('roberta.encoder.layer.{args.layer_id}.output.LayerNorm'))
	# method3 
	# model.roberta.pooler.register_forward_hook(get_activation("roberta.pooler"))
	# method4
	# model.heads.anli3[1].register_forward_hook(get_activation("heads.anli3[1]"))
	# method5
	# model.heads.anli1[4].register_forward_hook(get_activation("heads.anli1[4]"))
	# method6
	# model.heads.anli3[2].register_forward_hook(get_activation("heads.anli3[2]"))


	#store elements into datastore
	for i in tqdm(range(0, num_samples)):
		input_ids = torch.tensor(train_datasets[i]['input_ids']).view(1,-1).to(model.device)
		target_ids = torch.tensor(train_datasets[i]['labels']).view(-1).to(model.device)

		with torch.no_grad():
			#obtain context context representations
			if "attention_mask" in train_datasets[i].keys():
				attention_mask = torch.tensor(train_datasets[i]["attention_mask"]).to(model.device)
			else:
				attention_mask = None
			#changed
			if args.dataset == "copa":
				input_ids = input_ids.view(-1, args.max_seq_length)
			outputs = model(input_ids, attention_mask, labels=target_ids)

			## write context vectors
			### method1
			if args.dataset == "copa":
				embeds = activation[f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'][:,0].reshape(1,-1).cpu().numpy().astype(np.float16)
				dstore_keys[i, :] = embeds
			else:
				dstore_keys[i, :] = activation[f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'].squeeze(0)[0].cpu().numpy().astype(np.float16)
			### method3
			# dstore_keys[i, :] = activation[f'roberta.pooler'].squeeze(0).cpu().numpy().astype(np.float16)
			# method4
			# dstore_keys[i, :] = activation[f'heads.anli3[1]'].squeeze(0).cpu().numpy().astype(np.float16)
			# method5  #changed
			# tmp = activation[f'heads.anli1[4]'].squeeze(0).cpu().numpy().astype(np.float16)
			# tmp = softmax(tmp, axis=-1)
			# dstore_keys[i, :] = tmp
			# method6
			# dstore_keys[i, :] = activation[f'heads.anli3[2]'].squeeze(0).cpu().numpy().astype(np.float16)


			#write value/next word token  #changed
			dstore_vals[i, ] = target_ids[0].cpu().numpy().astype(np.intc)
			# dstore_vals[i, ] = target_ids.squeeze(0)[0].cpu().numpy().astype(np.intc)
	

	print(f"\n\n\t\tDatastore construction done! and saved to {dstore_filename}")
	print(f"\n\n\t\tStarting to build faiss index...")

	
	###### Build Faiss #####
	# Define the size of index vectors
	vector_dimension = hidden_dim

	# Create the Flat L2 index
	index = faiss.IndexFlatL2(vector_dimension)
	# res = faiss.StandardGpuResources()
	# index = faiss.IndexFlatL2(vector_dimension)
	# gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)

	# Add vectors to index. Embeddings can be Numpy arrays or torch tensors
	index.add(dstore_keys)
	# gpu_index_flat.add(dstore_keys)

	index_name = "./indexs/" + args.dataset + "_index_layer" + str(args.layer_id)

	faiss.write_index(index, index_name)
	# faiss.write_index(gpu_index_flat, index_name)

def faiss_read(dataset, layer_id):
	# Read the index from disk
	index_name = "./indexs/" + dataset + "_index_layer" + str(layer_id)

	# res = faiss.StandardGpuResources()
	index = faiss.read_index(index_name)
	# gpu_index = faiss.index_cpu_to_gpu(provider=res, device=0, index=index)

	# return gpu_index
	return index

def compute_accuracy(predictions, references):
	if isinstance(predictions, tuple):
		predictions = predictions[0]
	predictions = np.argmax(predictions, axis=1)
	metric = load_metric("accuracy")
	return metric.compute(predictions=predictions, references=references)

def compute_accuracy_(labels, predictions):
	preds = np.argmax(predictions, axis=-1)
	nums = len(labels)
	count = 0
	for i,(l, p) in enumerate(zip(labels, preds)):
		if (l==p):
			count += 1

	return round(count * 1.0/nums, 4)

def compute_macro_f1(labels, predictions):
	predictions = np.array(predictions).squeeze(1)
	preds = np.argmax(predictions, axis=-1)
	# label=[0,1] 
	p_weighted, r_weighted, f_weighted, support_weighted = precision_recall_fscore_support(labels, preds, average='macro')
	return round(f_weighted, 4)

def get_samples_classified_correctly_by_knnlm(test_datasets, y_true, y_lm_pre, y_knn_lm_pred):
	print("test_datasets keys:", test_datasets[0].keys())
	sample_idxs = []
	for i in range(len(y_true)):
		if y_true[i] == y_knn_lm_pred[i] and y_true[i]!=y_lm_pre[i]:
			sample_idxs.append(i)
	print("&&&&&&&&&&&&&&& sample_idxs &&&&&&&&&&&&&&&&&&&&&&")
	print(sample_idxs)
	for idx in sample_idxs:
		print(idx, " label: ", test_datasets[idx]["labels"], " ground truth: ", y_true[idx], " knn_lm: ", y_knn_lm_pred[idx], " lm_only: ", y_lm_pre[idx])

def get_test_acc(args, test_datasets, index, num_labels, model):
	y_true = []
	y_pred = []
	lm_logits = []
	neighbours_matrix=[]
	neighbour_labels_matrix=[]
	neighbours_distance_matrix=[]

	dstore_filename = args.dstore_path + args.dataset + "_dstore_" + str(args.layer_id)
	
	train_labels = np.memmap(dstore_filename + '_val.npy', dtype=np.int32, mode='r')
	# method1
	model.roberta.encoder.layer[args.layer_id].attention.output.LayerNorm.register_forward_hook(get_activation(f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'))

	knn_need_count=0
	search_time_sum = 0
	start_cm_time = time.time()
	for i, data in enumerate(test_datasets):
		if args.dataset == "copa":
			shape1 = 2
		else:
			shape1 = 1
		input_ids = torch.tensor(test_datasets[i]['input_ids']).view(shape1,-1).to(model.device)
		target_ids = torch.tensor(test_datasets[i]['labels']).view(-1).to(model.device)
		if "attention_mask" in data.keys():
			attention_mask = torch.tensor(data["attention_mask"]).to(model.device)
		else:
			attention_mask = None

		with torch.no_grad():
			#obtain context context representations
			if args.dataset == "copa":
				input_ids = input_ids.view(-1, args.max_seq_length)
			lm_output = model(input_ids, attention_mask, labels=target_ids)
			logits = torch.softmax(lm_output.logits, dim=-1)
			lm_logit = logits.detach().cpu().numpy()
			lm_logits.extend(lm_logit)
			
		
			# Search the top-k nearest neighbors of all the vectors in embedding
			# method1
			if args.dataset == "copa":
				embedding = activation[f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'][:,0,:].reshape(1,-1).squeeze(0).cpu().numpy().astype(np.float16)
			else:
				embedding = activation[f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'].squeeze(0)[0].cpu().numpy().astype(np.float16)
			
			embedding = np.expand_dims(embedding, axis=0)
			search_knnlm_start = time.time()
			distances, neighbours = index.search(embedding, k=512)   # shape: (1,4)

			neighbour_labels = []
			for idx in neighbours[0]:
				idx = idx.item()
				lab = train_labels[idx]
				neighbour_labels.append(lab)
			search_knnlm_end = time.time()

			search_time_sum += search_knnlm_end-search_knnlm_start

			y_lm_pred = torch.argmax(logits, dim=-1).cpu().tolist()

			y_true.append(target_ids.squeeze(0).cpu().tolist())
			y_pred.append(y_lm_pred)
			neighbours_matrix.append(neighbours[0].tolist())
			neighbour_labels_matrix.append(neighbour_labels)
			neighbours_distance_matrix.append(distances[0])
			

	end_cm_time = time.time()
	# lm_only_time = time.strftime('CM-only : %Y-%m-%d %H:%M:%S', time.localtime(end_cm_time-start_cm_time))
	# print(lm_only_time)
	print("lm only time needed:", end_cm_time-start_cm_time-search_time_sum)
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

				knnlm_start_time = time.time()

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
					knn_logits.append(knn_logit)

					# kl_divergence = sum(rel_entr(lm_logits[j], uniform_dis))
					kl_divergence = sum(kl_div(lm_logits[j], uniform_dis))
						
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

				knnlm_end_time = time.time()
				# knnlm_time=time.strftime('KNNLM : %Y-%m-%d %H:%M:%S', time.localtime(knnlm_end_time-knnlm_start_time))
				# print(knnlm_time)
				print("search time:", search_time_sum)
				print("knnlm time needed:", knnlm_end_time - knnlm_start_time + search_time_sum)

				# labels = [i for i in range(num_labels)]
				print("\n#num of under kl_threshold: ", knn_need_count, "\n")
				print("================ LM only======================")
				# print(classification_report(y_true, y_pred, labels=labels, digits=4))
				print(classification_report(y_true, y_pred, digits=4))

				lm_logits = np.stack(lm_logits,0)

				if args.dataset in ["anli", "anli1", "anli2", "anli3", "restaurant", "laptop", "rotten_tomatoes"]:
					metrics = compute_accuracy(lm_logits, y_true)
				else:
					raise NotImplementedError
				print(metrics)

				print("================ KNN only======================")
				# print(classification_report(y_true, y_knn_preds, labels=labels, digits=4))
				print(classification_report(y_true, y_knn_preds, digits=4))
				knn_logits = np.stack(knn_logits,0)

				if args.dataset in ["anli", "anli1", "anli2", "anli3", "restaurant", "laptop", "rotten_tomatoes"]:
					metrics = compute_accuracy(knn_logits, y_true)
				else:
					raise NotImplementedError
				print(metrics)

				print("================ KNN-LM ======================")
				# print(classification_report(y_true, y_knn_lm_pred, labels=labels, digits=4))
				print(classification_report(y_true, y_knn_lm_pred, digits=4))
				
				knn_lm_logits = np.stack(knn_lm_logits,0)
		
				if args.dataset in ["anli", "anli1", "anli2", "anli3", "restaurant", "laptop", "rotten_tomatoes"]:
					metrics = compute_accuracy(knn_lm_logits, y_true)
					tmp["metrics"] = metrics["accuracy"]
				else:
					raise NotImplementedError
				print(metrics)

				results.append(tmp)
				exp_idx = exp_idx + 1

	sorted_result = sorted(results, key=lambda k: k["metrics"], reverse=True)
	print(sorted_result[:10])

if __name__ == "__main__":

	args = get_args()

	if not os.path.exists(args.dstore_path):
		os.makedirs(args.dstore_path)

	'''Load pre-trained language model'''
	model = RobertaModelWithHeads.from_pretrained(args.model_id, output_hidden_states=True, )
	tokenizer = RobertaTokenizer.from_pretrained(args.model_id)

	if args.use_adapter:
		print("\n\n\tLoading adapter...\n")
		model.load_adapter(f"{args.adapter_path}")
		model.set_active_adapters(f"{args.dataset}") # to deactivate use model.set_active_adapters(None)

	# print(model)

	if args.num_proc==-1:
		args.num_proc = psutil.cpu_count()

	'''Load dataset'''
	if args.dataset.lower() == "rotten_tomatoes":

		infer = load_dataset("rotten_tomatoes", cache_dir="./Dataset/rotten_tomatoes/")
		infer = infer.rename_column("label", "labels")

		tokenized_datasets = infer.map(tokenize, batched=True,
			remove_columns=infer["train"].column_names,
			num_proc = args.num_proc)

		train_datasets = tokenized_datasets["train"]
		valid_datasets = tokenized_datasets["validation"]
		test_datasets = tokenized_datasets["test"]

	elif args.dataset.lower() == "trec":

		infer = load_dataset("trec", cache_dir="./Dataset/trec/")
		infer['train'] = infer['train'].shuffle(seed=42)
		dataset_tr_valid = infer['train'].train_test_split(test_size=0.1, shuffle=True, seed=100)
		infer['train'] = dataset_tr_valid['train']
		infer['validation'] = dataset_tr_valid['test']
		infer = infer.rename_column("coarse_label", "labels")

		tokenized_datasets = infer.map(tokenize, batched=True,
			remove_columns=infer["train"].column_names,
			num_proc = args.num_proc)

		train_datasets = tokenized_datasets["train"]
		valid_datasets = tokenized_datasets["validation"]
		test_datasets = tokenized_datasets["test"]

	elif args.dataset.lower() == "anli3":

		infer = load_dataset("anli", split=["train_r3","dev_r3","test_r3"], cache_dir="./Dataset/anli3/")
		infer[0] = infer[0].rename_column("label", "labels")
		infer[1] = infer[1].rename_column("label", "labels")
		infer[2] = infer[2].rename_column("label", "labels")

		train_datasets = infer[0].map(multi_input_tokenize, batched=True, remove_columns=infer[0].column_names, num_proc = args.num_proc)
		valid_datasets = infer[1].map(multi_input_tokenize, batched=True, remove_columns=infer[0].column_names, num_proc = args.num_proc)
		test_datasets = infer[2].map(multi_input_tokenize, batched=True, remove_columns=infer[0].column_names, num_proc = args.num_proc)

	elif args.dataset.lower() == "anli2":
		infer = load_dataset("anli", split=["train_r2","dev_r2","test_r2"], cache_dir="./Dataset/anli2/")
		infer[0] = infer[0].rename_column("label", "labels")
		infer[1] = infer[1].rename_column("label", "labels")
		infer[2] = infer[2].rename_column("label", "labels")

		train_datasets = infer[0].map(multi_input_tokenize, batched=True, remove_columns=infer[0].column_names, num_proc = args.num_proc)
		valid_datasets = infer[1].map(multi_input_tokenize, batched=True, remove_columns=infer[0].column_names, num_proc = args.num_proc)
		test_datasets = infer[2].map(multi_input_tokenize, batched=True, remove_columns=infer[0].column_names, num_proc = args.num_proc)

	elif args.dataset.lower() == "anli1":
		infer = load_dataset("anli", split=["train_r1","dev_r1","test_r1"], cache_dir="./Dataset/anli1/")
		infer[0] = infer[0].rename_column("label", "labels")
		infer[1] = infer[1].rename_column("label", "labels")
		infer[2] = infer[2].rename_column("label", "labels")

		train_datasets = infer[0].map(multi_input_tokenize, batched=True, remove_columns=infer[0].column_names, num_proc = args.num_proc)
		valid_datasets = infer[1].map(multi_input_tokenize, batched=True, remove_columns=infer[0].column_names, num_proc = args.num_proc)
		test_datasets = infer[2].map(multi_input_tokenize, batched=True, remove_columns=infer[0].column_names, num_proc = args.num_proc)

	elif args.dataset.lower() == "anli":
		num_of_labels = 3
		id_2_label={0:"0", 1:"1", 2:"2"}

		dataset = load_dataset("anli", cache_dir="./Dataset/anli/")
		dataset = dataset.rename_column("label", "labels")
		
		dataset = dataset.map(multi_input_tokenize, batched=True, remove_columns=dataset["train_r2"].column_names)
		dataset.set_format(type="torch", columns=["input_ids",  "labels"])
		
		train_datasets = torch.utils.data.ConcatDataset([dataset["train_r1"], dataset["train_r2"], dataset["train_r3"]])
		valid_datasets = torch.utils.data.ConcatDataset([dataset["dev_r1"], dataset["dev_r2"], dataset["dev_r3"]])
		test_datasets = torch.utils.data.ConcatDataset([dataset["test_r1"], dataset["test_r2"], dataset["test_r3"]])

	elif args.dataset.lower() == "restaurant":
		restaurant_file = "./Dataset/RLDS/Restaurants_Train_v2.csv"
		id_2_label={0:"conflict", 1:"negative", 2:"neutral", 3:"positive"}

		train_datasets = RLDSDataset(restaurant_file, tokenizer, 80, id_2_label, "train")
		valid_datasets = RLDSDataset(restaurant_file, tokenizer, 80, id_2_label, "valid")
		test_datasets = RLDSDataset(restaurant_file, tokenizer, 80, id_2_label, "test")

	elif args.dataset.lower() == "laptop":
		laptop_file = "./Dataset/RLDS/Laptop_Train_v2.csv"
		id_2_label={0:"conflict", 1:"negative", 2:"neutral", 3:"positive"}

		train_datasets = RLDSDataset(laptop_file, tokenizer, 90, id_2_label, "train")
		valid_datasets = RLDSDataset(laptop_file, tokenizer, 90, id_2_label, "valid")
		test_datasets = RLDSDataset(laptop_file, tokenizer, 90, id_2_label, "test")

	elif args.dataset.lower() == "device":
		root_dir = "./Dataset/DLRS/"

		id_2_label={0:"NEG", 1:"POS"}

		train_datasets = DLRSDataset(root_dir, "device", tokenizer, 400, id_2_label, mode="train")
		valid_datasets = DLRSDataset(root_dir, "device", tokenizer, 400, id_2_label, mode="valid")
		test_datasets = DLRSDataset(root_dir, "device", tokenizer, 400, id_2_label, mode="test")

	elif args.dataset.lower() == "service":
		root_dir = "./Dataset/DLRS/"

		id_2_label={0:"NEG", 1:"NEU", 2:"POS"}

		train_datasets = DLRSDataset(root_dir, "service", tokenizer, 350, id_2_label, mode="train")
		valid_datasets = DLRSDataset(root_dir, "service", tokenizer, 350, id_2_label, mode="valid")
		test_datasets = DLRSDataset(root_dir, "service", tokenizer, 350, id_2_label, mode="test")

	elif args.dataset.lower() == "rest":
		root_dir = "./Dataset/DLRS/"

		id_2_label={0:"NEG", 1:"NEU", 2:"POS"}

		train_datasets = DLRSDataset(root_dir, "rest", tokenizer, 350, id_2_label, mode="train")
		valid_datasets = DLRSDataset(root_dir, "rest", tokenizer, 350, id_2_label, mode="valid")
		test_datasets = DLRSDataset(root_dir, "rest", tokenizer, 350, id_2_label, mode="test")

	elif args.dataset.lower() == "new_laptop":
		root_dir = "./Dataset/DLRS/"

		id_2_label={0:"NEG", 1:"NEU", 2:"POS"}

		train_datasets = DLRSDataset(root_dir, "laptop", tokenizer, 430, id_2_label, mode="train")
		valid_datasets = DLRSDataset(root_dir, "laptop", tokenizer, 430, id_2_label, mode="valid")
		test_datasets = DLRSDataset(root_dir, "laptop", tokenizer, 430, id_2_label, mode="test")
	
	elif args.dataset.lower() in SUPERGLUE:
		dataset = SuperGlueDataset(tokenizer, args)
		num_of_labels = dataset.num_labels
		if args.dataset.lower() == "copa":
			id_2_label=None
		else:
			id_2_label = dataset.id2label
		train_datasets = dataset.train_dataset
		test_datasets = dataset.predict_dataset

		train_datasets = train_datasets.rename_column("label", "labels")
		test_datasets  = test_datasets.rename_column("label", "labels")

	elif args.dataset.lower() == "cb":
		train_dataset = load_dataset('super_glue', 'cb', split='train', cache_dir="./Dataset/super_glue/cb/")
		valid_dataset = load_dataset('super_glue', 'cb', split='validation', cache_dir="./Dataset/super_glue/cb/")
		test_dataset = valid_dataset

		train_dataset = train_dataset.rename_column("label", "labels")
		valid_dataset = valid_dataset.rename_column("label", "labels")
		test_dataset  = test_dataset.rename_column("label", "labels")

		# Encode the input data
		train_datasets = train_dataset.map(cb_encode_batch, batched=True, remove_columns=train_dataset.column_names)
		valid_datasets = valid_dataset.map(cb_encode_batch, batched=True, remove_columns=valid_dataset.column_names)
		test_datasets  = test_dataset.map(cb_encode_batch, batched=True, remove_columns=test_dataset.column_names)
	

	else:
		raise NotImplementedError

	if torch.cuda.is_available():
		model = model.cuda()

	
	# hidden_dim = model.config.hidden_size

	if args.create_dstore == True:
		'''Create Datastore and Build Faiss'''
		create_datastore(args, num_samples=len(train_datasets), train_datasets=train_datasets, model=model)

	'''Faiss Read Index'''
	
	index = faiss_read(args.dataset, args.layer_id)

	'''Test Dataset Acc'''
	# get_test_acc(args, valid_datasets, index, args.num_labels, model)
	get_test_acc(args, test_datasets, index, args.num_labels, model)
	







