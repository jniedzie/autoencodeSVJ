#!/bin/bash

cd /afs/cern.ch/work/j/jniedzie/private/svjets/autoencodeSVJ/rootToH5converter

echo "Activating conda"

source /usr/lib/python3.6/site-packages/conda/shell/etc/profile.d/conda.sh
conda activate ml

echo "Python version:"
python --version

echo "Setting up ROOT"

. /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.22.02/x86_64-centos7-gcc48-opt/bin/thisroot.sh

i=$(($1-1))

echo "Running for QCD part: ${i}"

#selections_type=all_cuts_fat_jets
#selections_type=all_cuts_ak4_jets
selections_type=no_lepton_veto_fat_jets
#selections_type=no_lepton_veto_ak4_jets

selections_path=../preselection/results/${selections_type}/qcd_merged/QCD_part_${i}.txt
output_path=results/${selections_type}/qcd/QCD_part_${i}.h5

python rootToH5.py -i $selections_path -o $output_path -e 3


#source /cvmfs/sft.cern.ch/lcg/views/LCG_93/x86_64-centos7-gcc7-opt/setup.sh
#
#python -c "import energyflow"
#if [ $? -eq 1 ]
#then
#    pip install energyflow --user
#fi
#
#env -i HOME=$HOME bash -i -c "source /afs/cern.ch/work/l/llepotti/private/CMS/CMSSW_8_0_20/src/autoencodeSVJ/conversion/setup.sh python /afs/cern.ch/work/l/llepotti/private/CMS/CMSSW_8_0_20/src/autoencodeSVJ/conversion/h5converter.py /afs/cern.ch/work/l/llepotti/private/CMS/CMSSW_8_0_20/src/autoencodeSVJ/data/background/eflow_3 /afs/cern.ch/work/l/llepotti/private/CMS/CMSSW_8_0_20/src/autoencodeSVJ/data/background/eflow_3/data_0_combined.txt data_0 0.8 100 -1 -1 0 3"
