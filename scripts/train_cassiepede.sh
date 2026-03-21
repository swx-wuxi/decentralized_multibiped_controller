export PYTHONPATH=.
export WANDB_API_KEY=
#### original training parameters
# --force_prob 0.2 \
# --poi_heading_range 1.05 \
# --num_cassie_prob 0.2 0.8 \
# --perturbation_force 30.0 \

python algo/cassiepede/trainer.py \
  --n_collectors 8 \
  --n_evaluators 0 \
  --time_horizon 300 \
  --max_steps 1000000 \
  --buffer_size 8000 \
  --eval_buffer_size 500 \
  --evaluate_freq 10 \
  --num_epoch 4 \
  --mini_batch_size 64 \
  --hidden_dim 64 \
  --lstm_hidden_dim 64 \
  --lstm_num_layers 2 \
  --use_orthogonal_init \
  --set_adam_eps \
  --use_adv_norm \
  --use_lr_decay \
  --use_grad_clip \
  --reward_name locomotion_cassiepede_feetairtime_modified \
  --project_name roadrunner_cassiepede \
  --device cpu \
  --position_offset 1.0 \
  --poi_heading_range 0.05 \
  --gamma 0.95 \
  --std 0.13 \
  --entropy_coef 0.01 \
  --num_cassie_prob 1.0 \
  --perturbation_force 0.0 \
  --force_prob 0.2 \
  --cmd_noise 0.0 0.0 0.0 \
  --cmd_noise_prob 0.0 \
  --wandb_mode online \
  --state_history_size 1 \
  --actor_name Actor_LSTM_v2 \
  --critic_name Critic_LSTM_v2 \
  --kl_check_min_itr 2 \
  --kl_check \
  --mirror_loss supervised