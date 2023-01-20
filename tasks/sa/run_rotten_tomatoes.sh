
# for epoch in 0 #2 3 4 5 6
# do
# 	CUDA_VISIBLE_DEVICES=2 python train.py --dataset rotten_tomatoes --epoch $epoch\
# 		--output_path ./save/training_output_rotten_tomatoes_$epoch \
# 		--save_adapter_path ./save/saved_adapters_rotten_tomatoes_$epoch
# done

# for layer_id in 11 #0 1 2 3 4 5 6 7 8 9 10 11
# do
# 	for epoch in 6 #2 3 4 5 6
# 	do
# 		CUDA_VISIBLE_DEVICES=2 python knn_lm.py --split train --dataset rotten_tomatoes --epoch $epoch \
# 		--model_id roberta-base --dstore_path /data/yingting/Dataset/rotten_tomatoes/dstore/$epoch/ \
# 		--use_adapter  --layer_id $layer_id --adapter_path ./save/saved_adapters_rotten_tomatoes_$epoch --num_labels 2 --create_dstore True
# 	done
# done

### for train.py
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset rotten_tomatoes --epoch 100 \
# 	--output_path ./save/training_output_rotten_tomatoes \
# 	--save_adapter_path ./save/saved_adapters_rotten_tomatoes


# for layer_id in 11 #0 1 2 3 4 5 6 7 8 9 10 11
# do
# 	CUDA_VISIBLE_DEVICES=3 python knn_lm.py --split train --dataset rotten_tomatoes \
# 	--model_id roberta-base --dstore_path /data/yingting/Dataset/rotten_tomatoes/dstore/ \
# 	--use_adapter  --layer_id $layer_id --adapter_path ./save/saved_adapters_rotten_tomatoes --num_labels 2 --create_dstore True
# done

#### for train1.py
# CUDA_VISIBLE_DEVICES=3 python train1.py --lr 1e-4 --batchsize 32 --dataset rotten_tomatoes --metric4train accuracy --epoch 100 \
# 	--output_path ./save/training_output_rotten_tomatoes --max_seq_length 80\
# 	--save_adapter_path ./save/saved_adapters_rotten_tomatoes

#### for knn_lm1.py
for layer_id in 11 #0 1 2 3 4 5 6 7 8 9 10 11
do
	CUDA_VISIBLE_DEVICES=2 python knn_lm1.py --split train --dataset rotten_tomatoes \
	--model_id roberta-base --dstore_path /data/yingting/Dataset/rotten_tomatoes/dstore/ \
	--use_adapter  --layer_id $layer_id --adapter_path ./save/saved_adapters_rotten_tomatoes --num_labels 2 #--create_dstore True
done

