#!/bin/bash


# run for all QCD parts from 1 to 30
#iConvs=($(seq 0 1 20))

# or pick just a few
iConvs=(0 2 3 4 5 6 7 8 9 10 11 13 14 15 16 17 18 19 20)

# list of lxplus servers that are working and can be accessed without providing a password
goodServers=(12 15 20 24 25 28 33 42 45 47 48 50 51 52 53 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70)

for iConv in "${iConvs[@]}"; do
  iServer=$((700+goodServers[$(($iConv))]))
  
  echo "Running for QCD part: ${iConv}"
  echo "Server: ${iServer}"
  
  # running the actual command
  (echo "Starting job $iConv on server lxplus$iServer"; ssh jniedzie@lxplus${iServer}.cern.ch -o "StrictHostKeyChecking no" -n -f "cd /afs/cern.ch/work/j/jniedzie/private/svjets/autoencodeSVJ/rootToH5converter; ./runConversionQCD.sh ${iConv}"; echo "Finished submitting job to server lxplus${iServer}") &
done
wait
echo All subshells finished
