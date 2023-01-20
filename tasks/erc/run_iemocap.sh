# CUDA_VISIBLE_DEVICES=1 python train_erc.py --dataset iemocap --lr 1e-3 --epochs 100 --batchsize 32 \
# 			--metric4train accuracy --max_seq_length 300 --output_path ./save/training_output_iemocap_1e3 \
# 			--save_adapter_path ./save/saved_adapters_iemocap_1e3

for layer_id in 11 #1 2 3 4 5 6 7 8 9 10 11
do
	CUDA_VISIBLE_DEVICES=1 python knnlm_erc.py --split train --dataset iemocap --max_seq_length 300 \
	--model_id roberta-base --dstore_path /data/yingting/Dataset/iemocap/dstore/ \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_iemocap_1e3 --num_labels 6 --use_adapter #--create_dstore True 
done