#!/bin/bash 
alias svjc="python $(pwd)/h5converter.py $@"
source /cvmfs/sft.cern.ch/lcg/views/LCG_93/x86_64-centos7-gcc7-opt/setup.sh
source $DELPHES_DIR/delphes-env.sh
echo "finished"
