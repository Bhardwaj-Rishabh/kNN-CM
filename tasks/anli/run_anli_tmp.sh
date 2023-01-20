for percent in 90 #90 80 70 60 50 40 30 20 10
do
	CUDA_VISIBLE_DEVICES=2 python knn_lm_fewshot.py --split train --dataset anli --fewshot_percent $percent \
	--model_id roberta-base --dstore_path ./Dataset/anli/dstore/fewshot_$percent- \
    --layer_id 11 --adapter_path ./save/saved_adapters_anli_fewshot_$percent --num_labels 3 --use_adapter --create_dstore True  >> anli_fewshot$percent-search.log
done