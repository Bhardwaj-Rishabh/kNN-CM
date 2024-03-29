CUDA_VISIBLE_DEVICES=2 python train.py --dataset rest --lr 1e-4 --epochs 100 --batchsize 32 \
		--output_path ./save/training_output_rest_tmp --save_adapter_path ./save/saved_adapters_rest_tmp


for layer_id in 11
do
	CUDA_VISIBLE_DEVICES=2 python knn_lm.py --split train --dataset rest \
	--model_id roberta-base --dstore_path ./Dataset/DLRS/dstore/ \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_rest_tmp --num_labels 3 --use_adapter --create_dstore True
done