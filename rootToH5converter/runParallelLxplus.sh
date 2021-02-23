#!/bin/bash

for iConv in {1..31}; do
  iServer=$((700+iConv))

  (echo "Starting job $iConv on server lxplus$iServer"; ssh jniedzie@lxplus${iServer}.cern.ch -o "StrictHostKeyChecking no" -n -f "cd /afs/cern.ch/work/j/jniedzie/private/svjets/autoencodeSVJ/rootToH5converter; ./runConversion.sh ${iConv}"; echo "Finished submitting job to server lxplus${iServer}") &
done
wait
echo All subshells finished
