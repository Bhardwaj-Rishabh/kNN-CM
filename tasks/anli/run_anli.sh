# CUDA_VISIBLE_DEVICES=3 python train.py --dataset anli --lr 1e-4 --epochs 100 --batchsize 32 \
# 			--metric4train accuracy --max_seq_length 300 --output_path ./save/training_output_anli \
# 			--save_adapter_path ./save/saved_adapters_anli


for layer_id in 11 #1 2 3 4 5 6 7 8 9 10 11
do
	CUDA_VISIBLE_DEVICES="" python knn_lm.py --split train --dataset anli \
	--model_id roberta-base --dstore_path ./Dataset/anli/dstore/ \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_anli --num_labels 3 --use_adapter #--create_dstore True 
done

# for layer_id in 0 #1 2 3 4 5 6 7 8 9 10
# do
# 	CUDA_VISIBLE_DEVICES=3 python knn_lm_valid.py --split train --dataset anli \
# 	--model_id roberta-base --dstore_path ./Dataset/anli/dstore/ \
#     --layer_id $layer_id --adapter_path ./save/saved_adapters_anli --num_labels 3 --use_adapter  --create_dstore True >> anli_valid_hyper_search_layer$layer_id.log  
# done

# for percent in  50 40 30 20 10 #90 80 70 60
# do
# 	CUDA_VISIBLE_DEVICES=2 python train_fewshot.py --dataset anli --lr 1e-4 --epochs 100 --batchsize 32 --fewshot_percent $percent\
# 			--metric4train accuracy --max_seq_length 300 --output_path ./save/training_output_anli_fewshot_$percent \
# 			--save_adapter_path ./save/saved_adapters_anli_fewshot_$percent >> anli_$percent.log
# done

# for percent in 80 70 60 #90 80 70 60 50 40 30 20 10
# do
# 	CUDA_VISIBLE_DEVICES=2 python knn_lm_fewshot.py --split train --dataset anli --fewshot_percent $percent \
# 	--model_id roberta-base --dstore_path ./Dataset/anli/dstore/fewshot_$percent- \
#     --layer_id 11 --adapter_path ./save/saved_adapters_anli_fewshot_$percent --num_labels 3 --use_adapter --create_dstore True  >> anli_fewshot$percent-search.log
# done