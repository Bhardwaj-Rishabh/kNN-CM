for lr in 3e-3 #1e-1 1e-2 1e-3 1e-4 1e-5
do
	for batchsize in 4 #2 4 8 16 32 64
	do
		CUDA_VISIBLE_DEVICES=0 python train1.py --lr $lr --epochs 100 --batchsize $batchsize --metric4train acc --max_seq_length 220 \
						--dataset cb --output_path ./save/training_output_cb_tmp --save_adapter_path ./save/saved_adapters_cb_tmp
	done
done

# 1e-3 4

# for layer_id in 11 #0 1 2 3 4 5 6 7 8 9 10 11
# do
# 	CUDA_VISIBLE_DEVICES=2 python knn_lm.py --split train --dataset cb \
# 	--pad_to_max_length True --max_seq_length 220 \
# 	--model_id roberta-base --dstore_path /data/yingting/Dataset/super_glue/cb/ \
#     --layer_id $layer_id --adapter_path ./save/saved_adapters_cb_tmp --num_labels 3 --use_adapter --create_dstore True
# done