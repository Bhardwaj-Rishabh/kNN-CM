# for lr in 1e-3 #1e-1 1e-2 1e-3 1e-4 1e-5
# do
# 	for batchsize in 32 #2 4 8 16 32 64
# 	do
# 		CUDA_VISIBLE_DEVICES=2 python train1.py --lr $lr --epochs 100 --batchsize $batchsize --metric4train accuracy --max_seq_length 850 \
# 					--dstore_path /data/yingting/Dataset/super_glue/boolq/ \
# 					--dataset boolq --output_path ./save/training_output_boolq --save_adapter_path ./save/saved_adapters_boolq
# 	done
# done


for layer_id in 11 #0 1 2 3 4 5 6 7 8 9 10 11
do
	CUDA_VISIBLE_DEVICES=2 python knn_lm.py --split train --dataset boolq --max_seq_length 850 \
	--model_id roberta-base --dstore_path /data/yingting/Dataset/super_glue/boolq/ \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_boolq --num_labels 2 --use_adapter #--create_dstore True
done

