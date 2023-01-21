export IN_TASK_NAME=anli
export DSTORE_PATH=./Dataset/$IN_TASK_NAME/dstore/
export OUT_TASK_NAME=cb

for layer_id in 11
do
	CUDA_VISIBLE_DEVICES=3 python domain_knnlm.py --split train --in_dataset $IN_TASK_NAME --out_dataset $OUT_TASK_NAME --max_seq_length 512 \
	--model_id roberta-base --dstore_path $DSTORE_PATH \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_$IN_TASK_NAME --num_labels 3 --create_dstore True --use_adapter --use_2dstores True #--create_dstore True
done