#!/bin/bash
set -e
cd /home/ubuntu/physics-sb-stage1
S1=runs/latent/s1_s1_scratch_64_b48_f128.pt
python3 exps/latent_stage.py --stage s2 --size 64 --steps 8000 --base_ch 48 --fdim 128 --s1_init $S1 --sigma_max 0.5
python3 exps/latent_stage.py --stage s2 --size 64 --steps 8000 --base_ch 48 --fdim 128 --s1_init $S1 --sigma_max 1.0
python3 exps/latent_stage.py --stage s2 --size 64 --steps 8000 --base_ch 48 --fdim 128 --s1_init $S1 --sigma_max 0.5 --no_freeze_prior
echo "S2 SWEEP DONE"
