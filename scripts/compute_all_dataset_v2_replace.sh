#!/bin/bash

for dataset in coco catsdogscars pascalpart monumai
do
    for network in CLIP-QDA CBM-classifier-logistic
    do
        for expl_method in LIME_CBM SHAP_CBM
        do
            echo "$dataset $network $expl_method"
            python make_samples.py --network $network --expl_method $expl_method --input_dataset $dataset 
        done
    done
done