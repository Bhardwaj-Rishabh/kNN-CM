CUDA_VISIBLE_DEVICES=3 python train.py --lr 1e-4 --batchsize 32 --dataset qasc --metric4train accuracy --epoch 100 \
	--output_path ./save/training_output_qasc --max_seq_length 30 \
	--save_adapter_path ./save/saved_adapters_qasc