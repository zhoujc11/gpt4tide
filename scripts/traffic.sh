seq_len=512
model=GPT4TS

for percent in 100
do
for pred_len in 96 192 336 720
do

python main.py \
    --root_path ./dataset/ \
    --data_path tide_single.csv \
    --model_id tide\
    --data custom \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --all 1 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model GPT4TS \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1

done
done

