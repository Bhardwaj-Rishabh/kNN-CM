###### train1.py
CUDA_VISIBLE_DEVICES=3 python train1.py --dataset restaurant --lr 1e-4 --epochs 100 --batchsize 32 \
	--dstore_path ./Dataset/RLDS/dstore/ \
	--metric4train accuracy  --output_path ./save/training_output_restaurant --save_adapter_path ./save/saved_adapters_restaurant
