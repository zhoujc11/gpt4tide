
# 定义四种策略
strategies=("lora" "frozen" "partial_wpe_ln" "full")

# 循环遍历策略
for strategy in "${strategies[@]}"
do
    echo "Running with strategy: $strategy"
    
    # 运行 Python 脚本并传递策略参数
    python /home/gpt4tide/main.py \
        --root_path /home/gpt4tide/dataset \
        --data_path chaowei.csv \
        --model_id tide \
        --data custom \
        --seq_len 256 \
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
        --patience 3 \
        --cos 1 \
        --tmax 10 \
        --is_gpt 1 \
        --is_tpe 1 \
        --is_cross 0 \
        --is_retrieval 0 \
        --do_visualize \
        --strategy "$strategy"  # 将策略作为参数传递
done
