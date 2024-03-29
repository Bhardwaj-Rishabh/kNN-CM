CUDA_VISIBLE_DEVICES=3 python train.py --dataset device -lr 1e-4 --epochs 100 --batchsize 32 \
		--metric4train accuracy --max_seq_length 300 \
		--output_path ./save/training_output_device --save_adapter_path ./save/saved_adapters_device


for layer_id in 11
do
	CUDA_VISIBLE_DEVICES=3 python knn_lm.py --split train --dataset device \
	--model_id roberta-base --dstore_path ./Dataset/DLRS/dstore/ \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_device --num_labels 2 --use_adapter --create_dstore True
done