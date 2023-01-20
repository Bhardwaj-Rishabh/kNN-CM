
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset trec --output_path ./save/training_output_trec --save_adapter_path ./save/saved_adapters_trec

for layer_id in 11 #0 1 2 3 4 5 6 7 8 9 10 11
do
	CUDA_VISIBLE_DEVICES=2 python knn_lm.py --split train --dataset trec \
	--model_id roberta-base --dstore_path /data/yingting/Dataset/trec/dstore/ \
	--use_adapter  --layer_id $layer_id --adapter_path ./save/saved_adapters_trec --num_labels 6 #--create_dstore True
done