#!/bin/bash
reg_strength=0.008


#run 106
python -u ssdr_main_S3DIS2.py --reg_strength ${reg_strength} --t 10000 --gpu 0 --round 2 --sampler T --point_uncertainty_mode sb --classbal 2 --uncertaint