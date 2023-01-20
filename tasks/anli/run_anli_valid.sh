
# for layer_id in 11 #1 2 3 4 5 6 7 8 9 10 11
# do
# 	CUDA_VISIBLE_DEVICES=2 python knn_lm_valid.py --split train --dataset anli \
# 	--model_id roberta-base --dstore_path /data/yingting/Dataset/anli/dstore/ \
#     --layer_id $layer_id --adapter_path ./save/saved_adapters_anli --num_labels 3 --use_adapter #--create_dstore True 
# done

for layer_id in 11 #1 2 3 4 5 6 7 8 9 10 11
do
	CUDA_VISIBLE_DEVICES=2 python knn_lm.py --split train --dataset anli \
	--model_id roberta-base --dstore_path /data/yingting/Dataset/anli/dstore/ \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_anli --num_labels 3 --use_adapter #--create_dstore True 
done