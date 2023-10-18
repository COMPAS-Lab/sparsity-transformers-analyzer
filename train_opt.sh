#!/bin/bash

python transformers/examples/pytorch/language-modeling/run_clm.py \
        --model_name_or_path "facebook/opt-350m" \
        --cache_dir "/chronos_data/tji/.huggingface_cache/transformers/opt350m-finetune-stancedet-abortion/" \
        --train_file "/chronos_data/tji/.huggingface_cache/datasets/twitter_stance/abortion_stance_postprocessing.csv" \
        --validation_split_percentage 5 \
        --output_dir "/chronos_data/tji/.huggingface_cache/transformers/opt350m-finetune-stancedet-abortion/" \
        --do_train \
        --do_eval \
        --per_device_train_batch_size 2 \
        --evaluation_strategy "steps" \
        --eval_steps 500 \
        --prediction_loss_only \
        --save_strategy "steps" \
        --save_steps 500 \
        --learning_rate 1e-5 \
        --num_train_epochs 120 \
        --warmup_steps 100 \
        --fp16 

