#!/bin/bash

#output_path=results/no_lepton_veto_fat_jets/qcd
#output_path=results/no_lepton_veto_ak4_jets/qcd
output_path=results/all_cuts_fat_jets/qcd/
#output_path=results/all_cuts_ak4_jets/qcd/

nEvents=$1
i=$2

cd /afs/cern.ch/work/j/jniedzie/private/svjets/autoencodeSVJ/preselection
. setenv.sh

mkdir -p $output_path

echo "Running QCD part: ${i}"

./SVJselection inputFileLists/qcd/input_file_list_qcd_${i}.txt QCD_part_${i} ${output_path} 0 $nEvents
