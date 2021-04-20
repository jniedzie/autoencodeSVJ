#!/bin/bash

input_path=$1
output_path=${input_path}_merged

mkdir -p $output_path

FILES=${input_path}/*selection*

index=0
new_index=0

for f in $FILES; do
	if [ $index -eq 50 ]
	then
		index=0
		new_index=$(($new_index+1))
	fi

	echo $f

	output_file_name=${output_path}/QCD_part_${new_index}.txt
	cat $f >> $output_file_name

	index=$(($index+1))

done;
