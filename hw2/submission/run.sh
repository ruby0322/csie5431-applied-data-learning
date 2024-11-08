
CUDA_VISIBLE_DEVICES=0,1 python summarization_inference.py \
    --model_name_or_path ./output/checkpoint-16284 \
    --do_predict \
    --test_file ${1} \
    --source_prefix "summarize: " \
    --text_column maintext \
    --output_dir ./preds/ \
    --output_file ${2} \
    --overwrite_output_dir \
    --predict_with_generate \
    --num_beams 4