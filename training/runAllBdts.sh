#!/bin/bash

# run for all signals from 1 to 30
iConvs=($(seq 1 1 30))

# or pick just a few
#iConvs=(12 18 20)

# list of lxplus servers that are working and can be accessed without providing a password

for iConv in "${iConvs[@]}"; do
  # here just figuring out what will be the mass and r_inv of the signal
  i=$(($iConv-1))
  ir=$(($i / 6))
  im=$(($i % 6))

  masses=(1500 2000 2500 3000 3500 4000)
  rinvs=(15 30 45 60 75)

  mass=${masses[$im]}
  rinv=${rinvs[$ir]}

  echo "Running for mass: ${mass}, r_inv: ${rinv}"
  echo "Server: ${iServer}"

  # running the actual command
  cd /Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/training
  (echo "Starting job $iConv"; python trainBdt.py -c configs.bdtDefaultConfig -i ${iConv};) &
done
wait
echo "All processes started"
