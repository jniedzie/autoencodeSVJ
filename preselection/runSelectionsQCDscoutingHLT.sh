#!/bin/bash

nEvents=$1
bin=$2


input_path=inputFileLists/qcd/scoutingHlt/eos/input_file_list_QCD_${bin}.txt

#output_path=results/no_lepton_veto_fat_jets/qcd
#output_path=results/no_lepton_veto_ak4_jets/qcd
output_path=results/all_cuts_fat_jets/qcd/scoutingHlt/eos/
#output_path=results/all_cuts_ak4_jets/qcd/

cd /afs/cern.ch/work/j/jniedzie/private/svjets/autoencodeSVJ/preselection
. setenv.sh

mkdir -p $output_path

echo "Running QCD part: ${i}"

./SVJselection ${input_path} QCD_${bin} ${output_path} 0 $nEvents
