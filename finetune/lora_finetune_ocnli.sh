formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deepspeed --include localhost:0 --master_port 19888 finetune.py \
    --model_name_or_path /content/MiniCPM3-4B \
    --output_dir output/OCNLILoRA/$formatted_time/ \
    --train_data_path data/data_6558/train.json \
    --eval_data_path data/data_6558/dev.json \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --model_max_length 1024 \
    --fp16 \
    --use_lora \
    --gradient_accumulation_steps 2 \
    --warmup_steps 200 \
    --max_steps 1000 \
    --weight_decay 0.01 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --seed 42 \
    --log_level info \
    --logging_strategy steps \
    --logging_steps 10 \
    --deepspeed configs/ds_config_zero2_offload.json