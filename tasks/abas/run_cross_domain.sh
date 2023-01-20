CUDA_VISIBLE_DEVICES=0 python knn_lm.py --split train --dataset rest2new_laptop \
	--model_id roberta-base --dstore_path /data/yingting/Dataset/DLRS/dstore/ \
    --layer_id 11 --adapter_path ./save/saved_adapters_rest --num_labels 3 --use_adapter #--create_dstore True