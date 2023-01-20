# for lr in 1e-3 #1e-1 1e-2 1e-3 1e-4 1e-5
# do
# 	for batchsize in 16 #2 4 8 16 32 64 128
# 	do
# 		echo "******************************"
# 		echo "	lr:$lr bs:$batchsize"
# 		echo "******************************"
# 		CUDA_VISIBLE_DEVICES=3 python train5.py --lr $lr --epochs 100 --batchsize $batchsize --metric4train accuracy --max_seq_length 512 \
# 						--dataset copa --output_path ./save/training_output_copa --save_adapter_path ./save/saved_adapters_copa
# 	done
# done

for layer_id in 11 #0 1 2 3 4 5 6 7 8 9 10 11
do
	CUDA_VISIBLE_DEVICES=3 python knnlm_superglue.py --split train --dataset copa --max_seq_length 512 \
	--model_id roberta-base --dstore_path /data/yingting/Dataset/super_glue/copa/ \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_copa --num_labels 1 --use_adapter --create_dstore True
done
