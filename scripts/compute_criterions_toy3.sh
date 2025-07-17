#!/bin/bash

for dataset in toy
do
    for sub_dataset in coco catsdogscars pascalpart monumai
    do
        for criterion in classification_logistic classification_qda classification_svm 
        do
            for network in resnet50-bcos vitB-bcos
            do
                echo "$dataset $sub_dataset $criterion $network "
                python compute_criterion.py --criterion $criterion --expl_method BCos --network $network --input_dataset $dataset --sub_dataset $sub_dataset
            done
        done
    done
done