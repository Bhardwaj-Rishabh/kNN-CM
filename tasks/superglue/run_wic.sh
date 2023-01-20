for lr in 1e-2 #1e-1 1e-2 1e-3 1e-4 1e-5
do
	for batchsize in 64 #2 4 8 16 32 64
	do
		CUDA_VISIBLE_DEVICES=3 python train.py --lr $lr --epochs 100 --batchsize $batchsize --metric4train accuracy --max_seq_length 60 \
						--dstore_path ./Dataset/super_glue/wic/ \
						--dataset wic --output_path ./save/training_output_wic --save_adapter_path ./save/saved_adapters_wic
	done
done

# for layer_id in 11 #0 1 2 3 4 5 6 7 8 9 10 11
# do
# 	CUDA_VISIBLE_DEVICES=0 python knn_lm.py --split train --dataset wic --max_seq_length 60 \
# 	--model_id roberta-base --dstore_path ./Dataset/super_glue/wic/ \
#     --layer_id $layer_id --adapter_path ./save/saved_adapters_wic --num_labels 2 --use_adapter --create_dstore True
# done
