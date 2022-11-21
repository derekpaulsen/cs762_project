#!/usr/bin/fish
#

set -l out ./out.json

for i in  (seq 10)
	# 50% orig
	python3.8 ./train_resnet.py --data_props '(.5, 1, 0, 0)' | tee -a $out 
	# 50% orig + 50% syn
	python3.8 ./train_resnet.py --data_props '(.5, 1, .5, 0)' | tee -a $out
	# 100% orig
	python3.8 ./train_resnet.py --data_props '(1, 1, 0, 0)' | tee -a $out
	# 100% syn
	python3.8 ./train_resnet.py --data_props '(0, 1, 1, 0)' | tee -a $out
	# 50% syn
	python3.8 ./train_resnet.py --data_props '(0, 1, .5, 0)' | tee -a $out
	# 100% orig + 100% syn
	python3.8 ./train_resnet.py --data_props '(1, 1, 1, 0)' | tee -a $out
end

