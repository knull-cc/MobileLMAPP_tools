formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time

export CUDA_VISIBLE_DEVICES=0

deepspeed --include localhost:0 --master_port 19888 finetune.py \
    --model_name_or_path /content/MiniCPM-2B-sft-bf16 \
    --output_dir output/OCNLILoRA/$formatted_time/ \
    --train_data_path data/MobileLMAPP/train.json \
    --eval_data_path data/MobileLMAPP/dev.json \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --model_max_length 1024 \
    --fp16 \
    --use_lora \
    --gradient_accumulation_steps 4 \
    --warmup_steps 100 \
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