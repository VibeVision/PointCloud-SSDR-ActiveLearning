#!/bin/bash
reg_strength=0.008
python -u partition/compute_superpoint --reg_strength ${reg_strength} > record_log/ssdr_log_sp.log 2>&1
python -u ssdr_create_seed.py --gpu 0 --seed_percent 0.005 --reg_strength ${reg_strength} > record_log/ssdr_log_seed_${reg_strength}.txt 2>&1
python -u ssdr_create_baseline.py --gpu 2 --dataset S3DIS --reg_strength ${reg_stre