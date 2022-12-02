#!/bin/bash
script='./train_vgg16.py'
# the directory where everything will be output
out_dir='/tmp'
props=('1,0,0' '.5,0,0' '.5,.5,0' '0,1,0' '0,.5,0' '1,1,0' '.5,0,.5' '0,0,1' '0,0,.5' '1,0,1' '1,1,1')

for i in 1 2 3 4 5
do
	for prop in ${props[@]}
	do
		## 100% orig BASELINE 
		out=$(mktemp -p $out_dir --suffix="_$(basename $script).json")
		echo "$out"
		CUDA_VISIBLE_DEVICES=4 python3.8 $script --data_props "$prop" | tee -a $out
	done
done

