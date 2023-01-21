# kNN-CM

## Motivation
Semi-parametric models exhibit the properties of both parametric and non-parametric modeling and have been shown to be effective in the next-word prediction language modeling task. However, there is a lack of studies on the text-discriminating properties of such models. We propose an inference-phase approach—k-Nearest Neighbor Classifier Model (kNN-CM)—that enhances the capacity of a pretrained parametric text classifier by incorporating a simple neighborhood search through the representation space of (memorized) training samples. The final class prediction of kNN-CM is based on the convex combination of probabilities obtained from kNN search
and prediction of the classifier. Our experiments show consistent performance improvements on eight SuperGLUE tasks, three adversarial natural language inference (ANLI)datasets, 11 question-answering (QA) datasets, and two sentiment classification datasets.

## Set up

```python
conda create --name knn python==3.8.5
conda activate knn

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch
pip install -U adapter-transformers
pip install datasets
pip install --upgrade psutil
pip install -U scikit-learn scipy matplotlib

# install Fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
# install FAISS
conda install -c pytorch faiss-gpu
```

## Train
### Train CM
```python
# train anli
CUDA_VISIBLE_DEVICES=3 python train.py --dataset anli --lr 1e-4 --epochs 100 --batchsize 32 \
 			--metric4train accuracy --max_seq_length 300 --output_path ./save/training_output_anli \
			--save_adapter_path ./save/saved_adapters_anli
```

### Create Datastore and inference on test dataset
```python
# create anli datastore
CUDA_VISIBLE_DEVICES=3 python knn_lm.py --split train --dataset anli \
	  --model_id roberta-base --dstore_path ./Dataset/anli/dstore/ \
    --layer_id 11 --adapter_path ./save/saved_adapters_anli --num_labels 3 --use_adapter --create_dstore True 
```
### Or can use .sh file directly
```python
# anli example
bash tasks/anli/run_anli.sh
```
