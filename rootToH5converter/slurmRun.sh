#!/bin/bash

#export KRB5CCNAME=DIR:/t3home/jniedzie/
#kinit jniedzie@CERN.CH -k -t /t3home/jniedzie/keytabs/jniedzie.keytab
#/usr/bin/eosfusebind -g krb5 ${HOME}/krb5cc_${UID}

selections_type=no_lepton_veto_fat_jets
#selections_type=no_lepton_veto_ak4_jets

efp_base=3
delta_r=0.8
delta_r_name=0p8
n_constituents=0
max_jets=2
verbosity=1
use_fat_jets=true
force_delphes_matching=true

output_path=""

if [[ $HOSTNAME == *"lxplus"* ]]; then
  echo "Setting paths for lxplus"
  output_path=/eos/cms/store/group/phys_exotica/svjets/s_channel_delphes/
else
  echo "Setting paths for ETH T3"
#  output_path=/pnfs/psi.ch/cms/trivcat/store/user/jniedzie/svjets/s_channel_delphes/
  output_path=results
fi

output_path=${output_path}/h5_${selections_type}_dr${delta_r_name}_efp${efp_base}_fatJets${use_fat_jets}_constituents${n_constituents}_maxJets${max_jets}/
#output_path=${output_path}/h5_${selections_type}_dr${delta_r_name}_efp${efp_base}_fatJets${use_fat_jets}_constituents${n_constituents}_maxJets${max_jets}/qcd

echo "Trying to create output directory"
echo "Output dir: ${output_path}"
mkdir -p $output_path

echo "Running conversion"


#masses=(1500 2000 2500 3000 3500 4000)
#rinvs=(30 50 70)

#masses=(4000 )
#rinvs=(30 50)
#
#for mass in "${masses[@]}"; do
#  for rinv in "${rinvs[@]}"; do
#    sbatch -p wn --account=t3 --job-name=rootToH5 --mem=4000M --time 02:00:00 -o output/%x-%j.out -e error/%x-%j.err runConversionSVJ.sh $mass $rinv
#  done
#done


# run for all QCD parts from 0 to 19
#iConvs=($(seq 0 1 19))

# or pick just a few
iConvs=(4 7 9 10 12 13 14 15 16 18)

for iConv in "${iConvs[@]}"; do
  sbatch -p wn --account=t3 --job-name=rootToH5 --mem=5000M --time 10:00:00 -o output/%x-%j.out -e error/%x-%j.err runConversionQCD.sh $iConv
done



