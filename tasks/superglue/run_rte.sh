for lr in 2e-4 #1e-1 1e-2 1e-3 1e-4 1e-5
do
	for batchsize in 32 #2 4 8 16 32 64
	do
		CUDA_VISIBLE_DEVICES=0 python train1.py --lr $lr --epochs 100 --batchsize $batchsize --metric4train accuracy --max_seq_length 250 \
						--dstore_path ./Dataset/super_glue/rte/ \
						--dataset rte --output_path ./save/training_output_rte_superglue_tmp --save_adapter_path ./save/saved_adapters_rte_superglue_tmp
	done
done

for layer_id in 11
do
	CUDA_VISIBLE_DEVICES=0 python knn_lm.py --split train --dataset rte --max_seq_length 250 \
	--model_id roberta-base --dstore_path ./Dataset/super_glue/rte/ \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_rte_superglue_tmp --num_labels 2 --use_adapter --create_dstore True
done