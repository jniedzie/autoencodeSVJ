#!/bin/bash

# run for all signals from 1 to 30
#iConvs=($(seq 0 1 29))

# or pick just a few
#iConvs=(0 1 2 3 4 5 6 7 8 9 10)
#iConvs=(11 12 13 14 15 16 17 18 19 20)
iConvs=(21 22 23 24 25 26 27 28 29 30)

# list of lxplus servers that are working and can be accessed without providing a password

for iConv in "${iConvs[@]}"; do
  cd /Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/training
  (echo "Starting job $iConv"; python train.py -c configs/bdt_default.py -i ${iConv};) &
done
wait
echo "All processes started"
