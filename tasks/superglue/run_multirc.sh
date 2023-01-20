# for lr in 1e-1 #1e-2 1e-3 1e-4 1e-5
# do
# 	for batchsize in 2 #4 8 16 32 64 128
# 	do
# 		echo "******************************"
# 		echo "	lr:$lr bs:$batchsize"
# 		echo "******************************"
# 		CUDA_VISIBLE_DEVICES=2 python train_superglue.py --lr $lr --epochs 100 --batchsize $batchsize --metric4train macro_f1 --max_seq_length 800 \
# 						--dataset multirc --output_path ./save/training_output_multirc --save_adapter_path ./save/saved_adapters_multirc
# 	done
# done

for layer_id in 11 #0 1 2 3 4 5 6 7 8 9 10 11
do
	CUDA_VISIBLE_DEVICES=1 python knnlm_superglue.py --split train --dataset multirc --max_seq_length 512 \
	--model_id roberta-base --dstore_path ./Dataset/super_glue/multirc/ \
    --layer_id $layer_id --num_labels 2 --use_adapter #--create_dstore True
done

