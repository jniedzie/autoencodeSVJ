#!/bin/bash

# run for all signals from 1 to 30
iConvs=($(seq 0 1 29))

# or pick just a few
#iConvs=(12 18 20)

# list of lxplus servers that are working and can be accessed without providing a password

for iConv in "${iConvs[@]}"; do
  cd /Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/training
  (echo "Starting job $iConv"; python train.py -c configs/bdt_default.py -i ${iConv};) &
done
wait
echo "All processes started"
