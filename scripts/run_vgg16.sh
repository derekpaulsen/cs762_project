#!/bin/bash
#


for i in seq 10
do
        # 50% orig
	    python ./train_vgg16.py --data_props '(.5, 1, 0, 0)' | tee -a out.json
	    # 50% orig + 50% syn
	    python ./train_vgg16.py --data_props '(.5, 1, .5, 0)' | tee -a out.json
	    # 100% orig
	    python ./train_vgg16.py --data_props '(1, 1, 0, 0)' | tee -a out.json
	    # 100% syn
	    python ./train_vgg16.py --data_props '(0, 1, 1, 0)' | tee -a out.json
	    # 50% syn
	    python ./train_vgg16.py --data_props '(0, 1, .5, 0)' | tee -a out.json
	    # 100% orig + 100% syn
	    python ./train_vgg16.py --data_props '(1, 1, 1, 0)' | tee -a out.json
done
