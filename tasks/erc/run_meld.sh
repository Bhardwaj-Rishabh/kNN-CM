CUDA_VISIBLE_DEVICES=2 python train_erc.py --dataset meld_e --lr 1e-5 --epochs 100 --batchsize 32 \
			--metric4train accuracy --max_seq_length 300 --output_path ./save/training_output_meld_e_1e5 \
			--save_adapter_path ./save/saved_adapters_meld_e_1e5

for layer_id in 11
do
	CUDA_VISIBLE_DEVICES=2 python knnlm_erc.py --split train --dataset meld_e --max_seq_length 300 \
	--model_id roberta-base --dstore_path ./Dataset/meld_e/dstore/ \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_meld_e --num_labels 6 --use_adapter --create_dstore True 
done