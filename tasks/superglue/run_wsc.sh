for lr in 1e-3 #1e-1 1e-2 1e-3 1e-4 1e-5
do
	for batchsize in 8 #2 4 8 16 32 64
	do

		CUDA_VISIBLE_DEVICES=0 python train.py --lr $lr --epochs 100 --batchsize $batchsize --metric4train f1 --max_seq_length 110 \
						--dataset wsc --output_path ./save/training_output_wsc --save_adapter_path ./save/saved_adapters_wsc
	done
done

# for layer_id in 11 #0 1 2 3 4 5 6 7 8 9 10 11
# do
# 	CUDA_VISIBLE_DEVICES=1 python knn_lm.py --split train --dataset wsc --max_seq_length 110 \
# 	--model_id roberta-base --dstore_path ./Dataset/super_glue/wsc/ \
#     --layer_id $layer_id --adapter_path ./save/saved_adapters_wsc --num_labels 2 --use_adapter --create_dstore True
# done
