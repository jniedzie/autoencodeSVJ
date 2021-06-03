#!/bin/bash

masses=(1500 2000 2500 3000 3500 4000)
#rinvs=(30 50 70)
#rinvs=(30 )
#rinvs=(50 )
rinvs=(70 )

for mass in "${masses[@]}"; do
  for rinv in "${rinvs[@]}"; do
    cd /Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/training
  (echo "Starting training for mass: $mass, rinv: $rinv"; python train.py -c configs/bdt_default.py -m ${mass} -r ${rinv};) &
  done
done
wait
echo "All processes started"
