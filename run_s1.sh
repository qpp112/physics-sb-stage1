#!/bin/bash
set -e
cd /home/ubuntu/physics-sb-stage1
AE=runs/latent/ae_64_b48_f128.pt
python3 exps/latent_stage.py --stage s1 --size 64 --steps 8000 --base_ch 48 --fdim 128 --ae_init $AE
python3 exps/latent_stage.py --stage s1 --size 64 --steps 8000 --base_ch 48 --fdim 128 --ae_init $AE --freeze_enc
python3 exps/latent_stage.py --stage s1 --size 64 --steps 8000 --base_ch 48 --fdim 128
echo "S1 SWEEP DONE"
