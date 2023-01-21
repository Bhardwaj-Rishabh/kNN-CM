
CUDA_VISIBLE_DEVICES=0 python train.py --dataset anli3 --lr 1e-3 --epochs 100 --batchsize 32 \
			--metric4train accuracy --max_seq_length 300 \
			--output_path ./save/training_output_anli3 --save_adapter_path ./save/saved_adapters_anli3

for layer_id in 11
do
	CUDA_VISIBLE_DEVICES=0 python knn_lm.py --split train --dataset anli3 \
	--model_id roberta-base --dstore_path ./Dataset/anli3/dstore/ \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_anli3 --num_labels 3 --use_adapter --create_dstore True 
done