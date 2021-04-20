#!/bin/bash


# run for all signals from 1 to 30
#iConvs=($(seq 1 1 30))

# or pick just a few
iConvs=(17 30)

# list of lxplus servers that are working and can be accessed without providing a password
goodServers=(1 3 4 6 7 8 10 11 12 13 14 16 18 19 20 21 23 25 26 27 28 30 31 32 33 34 35 36 37 38)

for iConv in "${iConvs[@]}"; do
  iServer=$((700+goodServers[$(($iConv-1))]))
  
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
  (echo "Starting job $iConv on server lxplus$iServer"; ssh jniedzie@lxplus${iServer}.cern.ch -o "StrictHostKeyChecking no" -n -f "cd /afs/cern.ch/work/j/jniedzie/private/svjets/autoencodeSVJ/rootToH5converter; ./runConversion.sh ${iConv}"; echo "Finished submitting job to server lxplus${iServer}") &
done
wait
echo All subshells finished
