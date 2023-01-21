CUDA_VISIBLE_DEVICES=3 python train1.py --lr 1e-4 --batchsize 32 --dataset rotten_tomatoes --metric4train accuracy --epoch 100 \
	--output_path ./save/training_output_rotten_tomatoes --max_seq_length 80\
	--save_adapter_path ./save/saved_adapters_rotten_tomatoes

for layer_id in 11 
do
	CUDA_VISIBLE_DEVICES=2 python knn_lm1.py --split train --dataset rotten_tomatoes \
	--model_id roberta-base --dstore_path ./Dataset/rotten_tomatoes/dstore/ \
	--use_adapter  --layer_id $layer_id --adapter_path ./save/saved_adapters_rotten_tomatoes --num_labels 2 --create_dstore True
done

