#!/bin/bash

echo "Running preselections"

nEvents=200000
#masses=(1500 2000 2500 3000 3500 4000)
masses=(1500)
rinvs=(30 50 70)

for mass in "${masses[@]}"; do
  for rinv in "${rinvs[@]}"; do

    sbatch -p quick --account=t3 --job-name=preselections --mem=3000M --time 00:30:00 -o output/%x-%j.out -e error/%x-%j.err runSelections.sh $nEvents $mass $rinv
  done
done



