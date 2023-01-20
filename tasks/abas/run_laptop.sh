# CUDA_VISIBLE_DEVICES=3 python train.py --dataset laptop --output_path ./save/training_output_laptop --save_adapter_path ./save/saved_adapters_laptop


for layer_id in 11 #0 1 2 3 4 5 6 7 8 9 10 11
do
	CUDA_VISIBLE_DEVICES=2 python knn_lm.py --split train --dataset laptop \
	--model_id roberta-base --dstore_path /data/yingting/Dataset/RLDS/dstore/ \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_laptop --num_labels 4 --use_adapter --create_dstore True 
done


# ####### train1.py
# CUDA_VISIBLE_DEVICES=1 python train1.py --dataset laptop --lr 1e-4 --epochs 100 --batchsize 32 \
# 	--dstore_path /data/yingting/Dataset/RLDS/dstore/ \
# 	--metric4train accuracy  --output_path ./save/training_output_laptop --save_adapter_path ./save/saved_adapters_laptop

