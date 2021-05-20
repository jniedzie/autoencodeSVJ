#!/bin/bash

file_name=$1
version=$2

sort -r -u ${file_name}*_v${version} > ${file_name}v${version}
