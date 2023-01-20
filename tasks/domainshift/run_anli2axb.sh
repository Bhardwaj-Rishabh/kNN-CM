#matthews_correlation

export IN_TASK_NAME=anli
export DSTORE_PATH=./Dataset/$IN_TASK_NAME/dstore/
export OUT_TASK_NAME=axb

for layer_id in 11 #0 1 2 3 4 5 6 7 8 9 10 11
do
	CUDA_VISIBLE_DEVICES=0 python domain_knnlm.py --split train --in_dataset $IN_TASK_NAME --out_dataset $OUT_TASK_NAME --max_seq_length 512 \
	--model_id roberta-base --dstore_path $DSTORE_PATH \
    --layer_id $layer_id --adapter_path ./save/saved_adapters_$IN_TASK_NAME --num_labels 3 --use_adapter #--create_dstore True
done