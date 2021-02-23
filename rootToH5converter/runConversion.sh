#!/bin/bash

cd /afs/cern.ch/work/j/jniedzie/private/svjets/autoencodeSVJ/rootToH5converter

echo "Activating conda"

#. /etc/profile.d/conda.sh
source /usr/lib/python3.6/site-packages/conda/shell/etc/profile.d/conda.sh

#eval "$(conda shell.bash hook)"
#conda init bash
conda activate ml

echo "Python version:"
python --version

echo "Setting up ROOT"

. /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.22.02/x86_64-centos7-gcc48-opt/bin/thisroot.sh

echo "Figuring out mass and r inv"

# find out sample name from the index
i=$(($1-1))
ir=$(($i / 6))
im=$(($i % 6))

masses=(1500 2000 2500 3000 3500 4000)
rinvs=(15 30 45 60 75)

mass=${masses[$im]}
rinv=${rinvs[$ir]}

echo "Running for mass: ${mass}, r_inv: ${rinv}"

#selections_type=all_cuts_fat_jets
#selections_type=all_cuts_ak4_jets
selections_type=no_lepton_veto_fat_jets
#selections_type=no_lepton_veto_ak4_jets

selections_path=../preselection/results/${selections_type}/SVJ_m${mass}_r${rinv}_selection.txt
output_path=results/${selections_type}/SVJ_m${mass}_r${rinv}.h5

python rootToH5.py -i $selections_path -o $output_path -e 3
