source /cvmfs/sft.cern.ch/lcg/views/LCG_93/x86_64-centos7-gcc7-opt/setup.sh

conda activate ml
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.22.02/x86_64-centos7-gcc48-opt/bin/thisroot.sh

python -c "import energyflow"
if [ $? -eq 1 ]
then 
    pip install energyflow --user
fi
