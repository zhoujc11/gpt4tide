
# 运行 Python 脚本并传递策略参数
python /home/gpt4tide/main.py \
    --root_path /home/gpt4tide/dataset \
    --data_path Hangzhou.csv \
    --model_id tide \
    --data custom \
    --seq_len 192 \
    --label_len 48 \
    --pred_len 96 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 50 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq '15min' \
    --patch_size 12 \
    --stride 6 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 1 \
    --model GPT4TS \
    --patience 10 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1 \
    --is_tpe 1 \
    --is_cross 1 \
    --is_retrieval 0 \
    --do_visualize \
    --strategy partial_wpe_ln  # 将策略作为参数传递

