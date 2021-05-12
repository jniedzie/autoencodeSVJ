#!/bin/bash

output_path=""

if [[ $HOSTNAME == *"lxplus"* ]]; then
  echo "Setting paths for lxplus"
  cd /afs/cern.ch/work/j/jniedzie/private/svjets/autoencodeSVJ/rootToH5converter
  output_path=/eos/cms/store/group/phys_exotica/svjets/s_channel_delphes/
  mkdir -p $output_path
else
  echo "Setting paths for ETH T3"
  cd /t3home/jniedzie/autoencodeSVJ/rootToH5converter
#  output_path=/pnfs/psi.ch/cms/trivcat/store/user/jniedzie/svjets/s_channel_delphes/
  output_path=results
fi

echo "Activating conda"

#source /usr/lib/python3.6/site-packages/conda/shell/etc/profile.d/conda.sh

eval "$(conda shell.bash hook)"
conda init bash
conda activate ml

echo "Python version:"
python --version

echo "Setting up ROOT"

#. /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.22.02/x86_64-centos7-gcc48-opt/bin/thisroot.sh

i=$(($1))

echo "Running for QCD part: ${i}"

#selections_type=all_cuts_fat_jets
#selections_type=all_cuts_ak4_jets
selections_type=no_lepton_veto_fat_jets
#selections_type=no_lepton_veto_ak4_jets

efp_base=3
delta_r=0.8
delta_r_name=0p8
n_constituents=150
max_jets=2
verbosity=1
use_fat_jets=true
force_delphes_matching=true

selections_path=../preselection/results/${selections_type}/qcd_merged_t3/QCD_part_${i}.txt

output_path=${output_path}/h5_${selections_type}_dr${delta_r_name}_efp${efp_base}_fatJets${use_fat_jets}_constituents${n_constituents}_maxJets${max_jets}/

output_path=${output_path}/qcd
output_path=${output_path}/QCD_part_${i}.h5

args=(
  -i "$selections_path"
  -o "$output_path"
  -e "$efp_base"
  -r "$delta_r"
  -c "$n_constituents"
  -j "$max_jets"
  -v "$verbosity"
)

if [[ -v use_fat_jets ]]; then
    args+=(-f)
fi

if [[ -v force_delphes_matching ]]; then
    args+=(-d)
fi

python rootToH5.py "${args[@]}"


#source /cvmfs/sft.cern.ch/lcg/views/LCG_93/x86_64-centos7-gcc7-opt/setup.sh
#
#python -c "import energyflow"
#if [ $? -eq 1 ]
#then
#    pip install energyflow --user
#fi
#
#env -i HOME=$HOME bash -i -c "source /afs/cern.ch/work/l/llepotti/private/CMS/CMSSW_8_0_20/src/autoencodeSVJ/conversion/setup.sh python /afs/cern.ch/work/l/llepotti/private/CMS/CMSSW_8_0_20/src/autoencodeSVJ/conversion/h5converter.py /afs/cern.ch/work/l/llepotti/private/CMS/CMSSW_8_0_20/src/autoencodeSVJ/data/background/eflow_3 /afs/cern.ch/work/l/llepotti/private/CMS/CMSSW_8_0_20/src/autoencodeSVJ/data/background/eflow_3/data_0_combined.txt data_0 0.8 100 -1 -1 0 3"
