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

#. /etc/profile.d/conda.sh
#source /usr/lib/python3.6/site-packages/conda/shell/etc/profile.d/conda.sh

eval "$(conda shell.bash hook)"
conda init bash
conda activate ml

echo "Python version:"
python --version

echo "Setting up ROOT"
#. /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.22.02/x86_64-centos7-gcc48-opt/bin/thisroot.sh
echo "Root version:"
root --version


#echo "Initializing kerberos and binding EOS..."
#kinit jniedzie@CERN.CH -k -t /t3home/jniedzie/keytabs/jniedzie.keytab
#/usr/bin/eosfusebind -g krb5 ${HOME}/krb5cc_${UID}
#echo "Done"


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

efp_base=3
delta_r=0.8
delta_r_name=0p8
n_constituents=150
max_jets=2
verbosity=1
use_fat_jets=true
force_delphes_matching=true

#selections_path=../preselection/results/${selections_type}/SVJ_m${mass}_r${rinv}_selection.txt
selections_path=../preselection/results/${selections_type}/SVJ_m3500_mDark20_r30_alphaPeak_selection.txt
#selections_path=../preselection/results/${selections_type}/SVJ_m3500_mDark40_r30_alphaPeak_selection.txt

echo "Using selections: ${selections_path}"



output_path=${output_path}/h5_${selections_type}_dr${delta_r_name}_efp${efp_base}_fatJets${use_fat_jets}_constituents${n_constituents}_maxJets${max_jets}/

#output_path=${output_path}/SVJ_m${mass}_r${rinv}.h5
output_path=${output_path}/SVJ_m3500_mDark20_r30_alphaPeak.h5
#output_path=${output_path}/SVJ_m3500_mDark40_r30_alphaPeak.h5

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
