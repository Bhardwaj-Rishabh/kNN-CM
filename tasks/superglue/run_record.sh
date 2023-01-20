# CUDA_VISIBLE_DEVICES=1 python train_superglue.py --lr 2e-4 --epochs 100 --batchsize 32 --metric4train f1 --max_seq_length 290 \
# 						--dataset record --output_path ./save/training_output_record --save_adapter_path ./save/saved_adapters_record


for layer_id in 11 #0 1 2 3 4 5 6 7 8 9 10 11
do
	CUDA_VISIBLE_DEVICES=0 python knnlm_superglue.py --split train --dataset record --max_seq_length 512 \
	--model_id roberta-base --dstore_path ./Dataset/super_glue/record/ \
    --layer_id $layer_id --num_labels 2 --use_adapter --create_dstore True
done



