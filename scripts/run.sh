#!/bin/bash
out='./out.json'
script='./train_resnet.py'

for i in 1 2 3 4 5
do
	# 100% orig BASELINE 
	python3.8 $script --data_props '(1, 0, 0)' | tee -a $out
	# 50% orig
	python3.8 $script --data_props '(.5, 0, 0)' | tee -a $out 


	# 50% orig + 50% syn
	python3.8 $script  --data_props '(.5, .5, 0)' | tee -a $out
	# 100% syn
	python3.8 $script --data_props '(0, 1, 0)' | tee -a $out
	# 50% syn
	python3.8 $script --data_props '(0, .5, 0)' | tee -a $out
	# 100% orig + 100% syn
	python3.8 $script --data_props '(1, 1, 0)' | tee -a $out


	# 50% orig + 50% syn2
	python3.8 $script  --data_props '(.5, 0, .5)' | tee -a $out
	# 100% syn2
	python3.8 $script --data_props '(0, 0, 1)' | tee -a $out
	# 50% syn2
	python3.8 $script --data_props '(0, 0, .5)' | tee -a $out
	# 100% orig + 100% syn2
	python3.8 $script --data_props '(1, 0, 1)' | tee -a $out


	# 100% orig + 100% syn + 100% syn2
	python3.8 $script --data_props '(1, 1, 1)' | tee -a $out
done

