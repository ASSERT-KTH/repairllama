python3 llama2_pred.py \
    --base_model_path <> \
    --lora_path <> \
    --data_path <> \
    --test_file <> \
    --output_file <> \
    --is_lora True \
    --max_length 512 \
    --max_new_tokens 256 \
    --do_sample True \
    --only_do_beam True \
    --only_do_topp False \
    --only_do_topk False \
    --only_do_temp False \
    --num_beams 10 \
    --temperature 0.8 \
    --top_k 0 \
    --top_p 0.95 \
    --request_num 10 \
