python odrl.py \
    --env='reacher' \
    --rl_on_real=False \
    --use_gpu=True \
    --log_dir='small_nodeltaR' \
    --real_reward_scaleup=1.0 \
    --num_epochs=300
