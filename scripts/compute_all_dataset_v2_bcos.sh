#!/bin/bash

for dataset in coco catsdogscars monumai
do
    for network in CLIP-QDA CBM-classifier-logistic
    do 
        for expl_method in Rise_CBM
        do
            echo "$dataset $network $expl_method"
            python make_samples.py --network $network --expl_method $expl_method --input_dataset $dataset 
        done
    done
done