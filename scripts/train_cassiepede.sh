export PYTHONPATH=.
export WANDB_API_KEY=

python algo/cassiepede/training.py \
  --n_collectors 4 \
  --n_evaluators 1 \
  --time_horizon 200 \
  --buffer_size 10000 \
  --eval_buffer_size 200 \
  --evaluate_freq 4 \
  --num_epoch 20 \
  --mini_batch_size 128 \
  --hidden_dim 64 \
  --lstm_hidden_dim 64 \
  --lstm_num_layers 2 \
  --use_orthogonal_init \
  --set_adam_eps \
  --kl_check \
  --kl_check_min_itr 2 \
  --use_adv_norm \
  --use_lr_decay \
  --use_grad_clip \
  --reward_name locomotion_cassiepede_feetairtime_modified \
  --project_name roadrunner_cassiepede \
  --wandb_mode offline \
  --device cuda:0 \
  --position_offset 1.0 \
  --poi_heading_range 1.05 \
  --gamma 0.95 \
  --std 0.13 \
  --entropy_coef 0.01 \
  --num_cassie_prob 0.2 0.8 \
  --wandb_mode online \
  --perturbation_force 30.0 \
  --force_prob 0.2 \
  --cmd_noise 0.0 0.0 0.0 \
  --cmd_noise_prob 0.0