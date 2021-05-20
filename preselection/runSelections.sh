#!/bin/bash

if [[ $HOSTNAME == *"lxplus"* ]]; then
  echo "Setting environment for lxplus"
  cd /afs/cern.ch/work/j/jniedzie/private/svjets/autoencodeSVJ/preselection
  . setenv.sh
else
  echo "Setting environemtn for eth T3"
  cd /t3home/jniedzie/autoencodeSVJ/preselection
fi

output_path=results/no_lepton_veto_fat_jets/
#output_path=results/no_lepton_veto_ak4_jets/
#output_path=results/all_cuts_fat_jets/
#output_path=results/all_cuts_ak4_jets/

mkdir -p $output_path

nEvents=$1
mass=$2
rinv=$3

echo "Running for mass: ${mass}, r_inv: ${rinv}"

#input_path=inputFileLists/input_file_list_m${mass}_r${rinv}.txt
input_path=inputFileLists/input_file_list_m${mass}_mDark20_r${rinv}_alphaPeak.txt
#sample_name=SVJ_m${mass}_r${rinv}
sample_name=SVJ_m${mass}_mDark20_r${rinv}_alphaPeak


./SVJselection $input_path $sample_name $output_path 0 $nEvents
