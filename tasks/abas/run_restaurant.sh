
CUDA_VISIBLE_DEVICES=3 python train1.py --dataset restaurant --lr 1e-3 --epochs 100 --batchsize 32 \
	--dstore_path ./Dataset/RLDS/dstore/ \
	--metric4train accuracy  --output_path ./save/training_output_restaurant --save_adapter_path ./save/saved_adapters_restaurant

for layer_id in 11
do  
	CUDA_VISIBLE_DEVICES=0 python knn_lm.py --split train --dataset restaurant \
	--model_id roberta-base --dstore_path ./Dataset/RLDS/dstore/ \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_restaurant --num_labels 4 --use_adapter --create_dstore True 
done


