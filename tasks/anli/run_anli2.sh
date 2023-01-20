
# CUDA_VISIBLE_DEVICES=2 python train.py --dataset anli2 --output_path ./save/training_output_anli2 --save_adapter_path ./save/saved_adapters_anli2

for layer_id in 11 #0 1 2 3 4 5 6 7 8 9 10 11
do
	CUDA_VISIBLE_DEVICES=1 python knn_lm.py --split train --dataset anli2 \
	--model_id roberta-base --dstore_path ./Dataset/anli2/dstore/ \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_anli2_1e3 --num_labels 3 --use_adapter #--create_dstore True #saved_adapters_anli2_1e3
done