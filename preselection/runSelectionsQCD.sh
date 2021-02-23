#!/bin/bash

nEvents=$1
i=$2

cd /afs/cern.ch/work/j/jniedzie/private/svjets/autoencodeSVJ/preselection
. setenv.sh

./SVJselection inputFileLists/qcd/input_file_list_qcd_${i}.txt QCD_part_${i} results/ 0 $nEvents
