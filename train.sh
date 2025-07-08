#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --chdir=.
export DATASET_ABBR=mixed-source
export MODEL_ABBR=scrn
export BERT_MODEL=klue/roberta-base # just used for huggingface wrapped model

python3 -u main.py  \
--do_train True \
--do_predict False \
--cache_dir .cache  \
--report_to "wandb" \
--seed 2020 \
--save_total_limit 5 \
--learning_rate 1e-5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_train_epochs 6.0 \
--max_seq_length 512 \
--num_labels 2 \
--logging_steps 50 \
--logging_strategy "steps" \
--evaluation_strategy "epoch" \
--save_strategy "epoch" \
--gradient_accumulation_steps 1 \
--metric_base_model_name_or_path gpt2 \
--model_name_or_path ${BERT_MODEL} \
--data_files /home/jiseung/TCN/data \
--output_dir ./data_out/${MODEL_ABBR}
