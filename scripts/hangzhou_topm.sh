for topm in {0..10}
do
    echo "------------------------------------------------------------------------------------------------------------ topm = $topm"
    python /home/time/GPT4TS-main/Long-term_Forecasting/main.py \
        --root_path /home/time/GPT4TS-main/Long-term_Forecasting/dataset \
        --data_path Hangzhou.csv \
        --model_id tide \
        --data custom \
        --seq_len 192 \
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
        --topm $topm  # Loop through topm from 0 to 10
done
