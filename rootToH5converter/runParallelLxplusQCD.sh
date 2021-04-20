#!/bin/bash


# run for all QCD parts from 1 to 30
#iConvs=($(seq 0 1 20))

# or pick just a few
iConvs=(10 11 17 20)

# list of lxplus servers that are working and can be accessed without providing a password
goodServers=(1 3 4 6 7 8 10 11 12 13 14 15 16 18 19 20 21 22 23 24 25 26 27 28 30 31 32 33 34 35)

for iConv in "${iConvs[@]}"; do
  iServer=$((700+goodServers[$(($iConv))]))
  
  echo "Running for QCD part: ${iConv}"
  echo "Server: ${iServer}"
  
  # running the actual command
  (echo "Starting job $iConv on server lxplus$iServer"; ssh jniedzie@lxplus${iServer}.cern.ch -o "StrictHostKeyChecking no" -n -f "cd /afs/cern.ch/work/j/jniedzie/private/svjets/autoencodeSVJ/rootToH5converter; ./runConversionQCD.sh ${iConv}"; echo "Finished submitting job to server lxplus${iServer}") &
done
wait
echo All subshells finished
