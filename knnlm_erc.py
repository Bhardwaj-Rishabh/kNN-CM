import os
import numpy as np

import torch
from tqdm import tqdm
from transformers import set_seed, RobertaTokenizer, RobertaModelWithHeads

import faiss
import psutil

from datasets import load_metric

from sklearn.metrics import classification_report
from collections import Counter


import argparse
from scipy.special import rel_entr
from scipy.special import kl_div

from train_erc import ERCDataset

set_seed(1234)
import random
random.seed(4)
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
	parser.add_argument('--lambdas', type=list, default=[0.1]) #1e-3, 1e-2, 0.05, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
	parser.add_argument('--topn', type=list, default=[32]) #1, 2, 4, 8, 16, 32, 64, 128, 256, 512
	parser.add_argument('--kl_thresholds', type=list, default=[0.9]) #0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8
	parser.add_argument('--pad_to_max_length', type=bool, default=True)
	parser.add_argument('--max_seq_length', type=int, default=128)

	args = parser.parse_args()

	return args

#register hook to extract input to final ffl after layer norm
activation = {}
def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook

from scipy.special import softmax
def create_datastore(args, num_samples, train_datasets, model):
	'''Allocate datastore memory'''	
	dstore_filename = args.dstore_path + args.dataset + "_dstore_" + str(args.layer_id)
	hidden_dim = model.config.hidden_size

	#create datastore
	dstore_keys = np.memmap(dstore_filename + '_key.npy', dtype=np.float16, mode='w+', shape=(num_samples, hidden_dim))
	dstore_vals = np.memmap(dstore_filename + '_val.npy', dtype=np.int32, mode='w+', shape=(num_samples, 1))

	'''Construct datastore'''
	# method1
	model.roberta.encoder.layer[args.layer_id].attention.output.LayerNorm.register_forward_hook(get_activation(f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'))

	#store elements into datastore
	for i in tqdm(range(0, num_samples)):
		input_ids = train_datasets[i]['input_ids'].view(1,-1).to(model.device)
		target_ids = train_datasets[i]['labels'].view(-1).to(model.device)

		with torch.no_grad():
			#obtain context context representations
			if "attention_mask" in train_datasets[i].keys():
				attention_mask = train_datasets[i]["attention_mask"].to(model.device)
			else:
				attention_mask = None
			outputs = model(input_ids, attention_mask, labels=target_ids)

			## write context vectors
			### method1
			dstore_keys[i, :] = activation[f'roberta.encoder.layer.{args.layer_id}.attention.output.LayerNorm'].squeeze(0)[0].cpu().numpy().astype(np.float16)

			#write value/next word token  #changed
			dstore_vals[i, ] = target_ids[0].cpu().numpy().astype(np.intc)
	
	print(f"\n\n\t\tDatastore construction done! and saved to {dstore_filename}")
	print(f"\n\n\t\tStarting to build faiss index...")

	
	###### Build Faiss #####
	# Define the size of index vectors
	vector_dimension = hidden_dim

	# Create the Flat L2 index
	index = faiss.IndexFlatL2(vector_dimension)

	# Add vectors to index. Embeddings can be Numpy arrays or torch tensors
	index.add(dstore_keys)

	index_name = "./indexs/" + args.dataset + "_index_layer" + str(args.layer_id)

	faiss.write_index(index, index_name)

def faiss_read(dataset, layer_id):
	# Read the index from disk
	index_name = "./indexs/" + dataset + "_index_layer" + str(layer_id)

	index = faiss.read_index(index_name)

	return index

def compute_accuracy(predictions, references):
	if isinstance(predictions, tuple):
		predictions = predictions[0]
	predictions = np.argmax(predictions, axis=1)
	metric = load_metric("accuracy")
	return metric.compute(predictions=predictions, references=references)

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
	for i, data in enumerate(test_datasets):
		input_ids = torch.tensor(test_datasets[i]['input_ids']).view(1,-1).to(model.device)
		target_ids = torch.tensor(test_datasets[i]['labels']).view(-1).to(model.device)
		if "attention_mask" in data.keys():
			attention_mask = data["attention_mask"].to(model.device)
		else:
			attention_mask = None

		with torch.no_grad():
			#obtain context context representations
			lm_output = model(input_ids, attention_mask, labels=target_ids)
			logits = torch.softmax(lm_output.logits, dim=-1)
			lm_logit = logits.detach().cpu().numpy()
			lm_logits.extend(lm_logit)
			
		
			# Search the top-k nearest neighbors of all the vectors in embedding
			# method1
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

				print("\n#num of under kl_threshold: ", knn_need_count, "\n")
				print("================ LM only======================")
				print(classification_report(y_true, y_pred, digits=4))

				lm_logits = np.stack(lm_logits,0)

				if args.dataset in ['dyda_e', 'iemocap', 'meld_e']:
					metrics = compute_accuracy(lm_logits, y_true)
				else:
					raise NotImplementedError
				print(metrics)

				print("================ KNN only======================")
				print(classification_report(y_true, y_knn_preds, digits=4))
				knn_logits = np.stack(knn_logits,0)

				if args.dataset in ['dyda_e', 'iemocap', 'meld_e']:
					metrics = compute_accuracy(knn_logits, y_true)
				else:
					raise NotImplementedError
				print(metrics)

				print("================ KNN-LM ======================")
				print(classification_report(y_true, y_knn_lm_pred, digits=4))
				
				knn_lm_logits = np.stack(knn_lm_logits,0)
		
				if args.dataset in ['dyda_e', 'iemocap', 'meld_e']:
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
	train_dataset = ERCDataset(args,"train", tokenizer)
	valid_dataset = ERCDataset(args,"valid", tokenizer)
	test_dataset = ERCDataset(args,"test", tokenizer)

	model = model.cuda()

	
	# hidden_dim = model.config.hidden_size

	if args.create_dstore == True:
		'''Create Datastore and Build Faiss'''
		create_datastore(args, num_samples=len(train_dataset), train_datasets=train_dataset, model=model)

	'''Faiss Read Index'''
	index = faiss_read(args.dataset, args.layer_id)

	'''Test Dataset Acc'''
	# get_test_acc(args, valid_dataset, index, args.num_labels, model)
	get_test_acc(args, test_dataset, index, args.num_labels, model)


