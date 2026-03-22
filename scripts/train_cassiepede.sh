export PYTHONPATH=.
export WANDB_API_KEY= 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
#### original training parameters
# --force_prob 0.2 \
# --poi_heading_range 1.05 \
# --num_cassie_prob 0.2 0.8 \
# --perturbation_force 30.0 \

# bufefer_size: 采样采集多少数据 （你每一次“更新模型”之前，到底看了多少真实经验）
python algo/cassiepede/trainer.py \
  --n_collectors 12 \
  --n_evaluators 0 \
  --time_horizon 500 \
  --max_steps 1000000 \
  --buffer_size 10000 \
  --eval_buffer_size 1000 \
  --evaluate_freq 4 \
  --num_epoch 5 \
  --mini_batch_size 32 \
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
  --poi_position_offset 1.5 \
  --poi_heading_range 1.05 \
  --gamma 0.95 \
  --std 0.13 \
  --entropy_coef 0.01 \
  --num_cassie_prob 1 1 1 \
  --perturbation_force 50.0 \
  --perturbation_torque 25.0 \
  --force_prob 0.1 \
  --cmd_noise 0.0 0.0 0.0 \
  --cmd_noise_prob 0.0 \
  --wandb_mode online \
  --state_history_size 1 \
  --actor_name Actor_LSTM_v2 \
  --critic_name Critic_LSTM_v2 \
  --kl_check_min_itr 2 \
  --kl_check \
  --mirror_loss supervised