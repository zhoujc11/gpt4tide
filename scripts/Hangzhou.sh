#!/bin/bash

# 数据集配置
DATA_PATH="Hangzhou.csv"

echo "===== 无cross，无tpe，无检索 ====="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 0 \
    --is_cross 0 \
    --is_retrieval 0 \
    --topm 5

echo "===== cross+无tpe+无检索 ====="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 0 \
    --is_cross 1 \
    --is_retrieval 0 \
    --topm 5

echo "===== cross+无tpe+检索topm=5 ====="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 0 \
    --is_cross 1 \
    --is_retrieval 1 \
    --topm 5

echo "===== cross+tpe+检索topm=5 ====="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 1 \
    --is_cross 1 \
    --is_retrieval 1 \
    --topm 5

echo "===== 无cross+tpe+无检索topm=5 ====="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 1 \
    --is_cross 0 \
    --is_retrieval 0 \
    --topm 5

echo "===== cross+tpe+无检索topm=5 ====="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 1 \
    --is_cross 1 \
    --is_retrieval 0 \
    --topm 5

echo "===== cross+tpe+检索topm=2 ====="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 1 \
    --is_cross 1 \
    --is_retrieval 1 \
    --topm 2

echo "===== cross+tpe+检索topm=4 ====="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 1 \
    --is_cross 1 \
    --is_retrieval 1 \
    --topm 4

echo "===== cross+tpe+检索topm=6 ====="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 1 \
    --is_cross 1 \
    --is_retrieval 1 \
    --topm 6

echo "===== cross+tpe+检索topm=8 ====="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 1 \
    --is_cross 1 \
    --is_retrieval 1 \
    --topm 8

echo "===== cross+tpe+检索topm=5+seqlen=256 ====="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 256 \
    --label_len 48 \
    --pred_len 96 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 1 \
    --is_cross 1 \
    --is_retrieval 1 \
    --topm 5

echo "===== 无cross+无tpe+无检索topm=5+seqlen=256 ====="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 256 \
    --label_len 48 \
    --pred_len 96 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 0 \
    --is_cross 0 \
    --is_retrieval 0 \
    --topm 5

    echo "===== 无cross+无tpe+无检索topm=5+seqlen=256 =====预测48=========="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 256 \
    --label_len 48 \
    --pred_len 48 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 0 \
    --is_cross 0 \
    --is_retrieval 0 \
    --topm 5


    echo "===== 无cross+无tpe+无检索topm=5+seqlen=96 =====预测48=========="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 48 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 0 \
    --is_cross 0 \
    --is_retrieval 0 \
    --topm 5

    echo "===== 无cross+无tpe+无检索topm=5+seqlen=48 =====预测48=========="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 48 \
    --label_len 24 \
    --pred_len 48 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 0 \
    --is_cross 0 \
    --is_retrieval 0 \
    --topm 5


    echo "===== cross+tpe+topm=5+seqlen=256 =====预测48=========="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 256 \
    --label_len 48 \
    --pred_len 48 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 1 \
    --is_cross 1 \
    --is_retrieval 1 \
    --topm 5


    echo "===== cross+tpe+topm=5+seqlen=96 =====预测48=========="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 48 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 1 \
    --is_cross 1 \
    --is_retrieval 1 \
    --topm 5

    echo "===== cross+tpe+topm=5+seqlen=48 =====预测48=========="
python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
    --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
    --data_path $DATA_PATH \
    --model_id tide \
    --data custom \
    --seq_len 48 \
    --label_len 24 \
    --pred_len 48 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 1 \
    --is_cross 1 \
    --is_retrieval 1 \
    --topm 5