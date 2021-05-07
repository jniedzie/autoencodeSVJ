#!/bin/bash


# run for all signals from 1 to 30
iConvs=($(seq 1 1 30))

# or pick just a few
#iConvs=(13 2 26 27 11 17 23 29 12)

# list of lxplus servers that are working and can be accessed without providing a password
goodServers=(61 62 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 83 84 85 86 87 90 91 92 93 94 95 96 97 98)

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
