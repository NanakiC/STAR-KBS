export CUDA_VISIBLE_DEVICES=0


model_name=iTransformer

seq_len=720

w_align=0.2
w_exo=1
dropout=0.1
use_star=1
noise_prob=0.9
for pred_len in 96 192 336 720
do
    echo "=================================================================================="
    echo "training $model_name,  (pred_len): $pred_len"
    echo "=================================================================================="
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path traffic.csv \
      --model_id traffic_$seq_len'_'$pred_len \
      --model $model_name \
      --data Traffic \
      --features MS \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --d_model 512 \
      --d_ff 2048 \
      --batch_size 16 \
      --train_epochs 30 \
      --learning_rate 0.001 \
      --dropout $dropout \
      --w_align $w_align \
      --w_exo $w_exo \
      --use_star  $use_star \
      --noise_prob $noise_prob \
      --pos 0 \
      --itr 1
done

