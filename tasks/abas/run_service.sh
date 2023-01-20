# CUDA_VISIBLE_DEVICES=3 python train.py --dataset service --output_path ./save/training_output_service --save_adapter_path ./save/saved_adapters_service


for layer_id in 11 #0 1 2 3 4 5 6 7 8 9 10 11
do
	CUDA_VISIBLE_DEVICES=3 python knn_lm.py --split train --dataset service \
	--model_id roberta-base --dstore_path /data/yingting/Dataset/DLRS/dstore/ \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_service --num_labels 3 --use_adapter #--create_dstore True
done