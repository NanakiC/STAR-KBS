export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

seq_len=720

w_align=0.1
w_exo=1
dropout=0.2
use_star=1
noise_prob=0.8
for pred_len in 96 168 336 720
do
    echo "=================================================================================="
    echo "training $model_name,  (pred_len): $pred_len"
    echo "=================================================================================="
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path weather.csv \
      --model_id weather_$seq_len'_'$pred_len \
      --model $model_name \
      --data Weather \
      --features MS \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --d_model 128 \
      --d_ff 256 \
      --dropout $dropout \
      --w_align $w_align \
      --noise_prob $noise_prob \
      --use_star  $use_star \
      --layer_norm 0 \
      --itr 1 
done


