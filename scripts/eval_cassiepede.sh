export PYTHONPATH=.
export WANDB_API_KEY="<your wandb api key goes here but not required for offline mode>"
# --runs_name "2024-05-01 10_30_51.809112" \
# --runs_name "2026-03-21 18:28:34.137934" \
python algo/cassiepede/evaluation.py \
  --seed 0 \
  --time_horizon 1000 \
  --project_name roadrunner_cassiepede \
  --num_cassie_prob 1 0\
  --position_offset 1.0 \
  --poi_heading_range 1.05 \
  --poi_position_offset 1.5 \
  --reward_name locomotion_cassiepede_feetairtime_modified \
  --perturbation_force 0.0 \
  --perturbation_torque 0.0 \
  --force_prob 0.0 \
  --runs_name "2026-03-22 19:04:34.532800" \
  --model_checkpoint latest \
  --evaluation_mode "interactive" \
  --state_history_size 1 \
  --offline \

