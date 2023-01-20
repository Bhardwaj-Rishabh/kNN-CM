# CUDA_VISIBLE_DEVICES=3 python train_erc.py --dataset dyda_e --lr 1e-4 --epochs 100 --batchsize 32 \
# 			--metric4train accuracy --max_seq_length 300 --output_path ./save/training_output_dyda_e \
# 			--save_adapter_path ./save/saved_adapters_dyda_e

for layer_id in 11 #1 2 3 4 5 6 7 8 9 10 11
do
	CUDA_VISIBLE_DEVICES=3 python knnlm_erc.py --split train --dataset dyda_e --max_seq_length 300 \
	--model_id roberta-base --dstore_path ./Dataset/dyda_e/dstore/ \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_dyda_e --num_labels 6 --use_adapter #--create_dstore True 
done