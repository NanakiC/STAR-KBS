export CUDA_VISIBLE_DEVICES=0


model_name=iTransformer

seq_len=720

dropout=0.2
w_align=0.1
use_star=1
noise_prob=0.6
for pred_len in 96 192 336 720
do
    echo "=================================================================================="
    echo "training $model_name,  (pred_len): $pred_len"
    echo "=================================================================================="
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --model_id ETTm2_$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features MS \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 128 \
      --d_ff 128 \
      --dropout $dropout \
      --w_align $w_align \
      --learning_rate 0.0005 \
      --noise_prob $noise_prob \
      --use_star  $use_star \
      --itr 1
done
