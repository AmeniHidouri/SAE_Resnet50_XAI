#!/bin/bash

for dataset in toy
do
    for sub_dataset in catsdogscars pascalpart
    do
        for criterion in variance_gauss variance_sharp variance_bright complexity_10 complexity_20 complexity_30 classification_logistic classification_qda classification_svm 
        do
            for network in CLIP-QDA CBM-classifier-logistic
            do 
                echo "$dataset $sub_dataset $criterion CLIP-QDA Rise_CBM"
                python compute_criterion.py --criterion $criterion --expl_method Rise_CBM --network $network --input_dataset $dataset --sub_dataset $sub_dataset
            done

            for expl_method in CLIP-QDA-sample LIME_CBM SHAP_CBM
            do
                echo "$dataset $sub_dataset $criterion CLIP-QDA $expl_method"
                python compute_criterion.py --criterion $criterion --expl_method $expl_method --network CLIP-QDA --input_dataset $dataset --sub_dataset $sub_dataset
            done

            for expl_method in LIME_CBM SHAP_CBM
            do
                echo "$dataset $sub_dataset $criterion CBM-classifier-logistic $expl_method"
                python compute_criterion.py --criterion $criterion --expl_method $expl_method --network CBM-classifier-logistic --input_dataset $dataset --sub_dataset $sub_dataset
            done

            echo "$dataset $sub_datasetXNES-classifier-logistic Xnesyl-Linear"
            python compute_criterion.py --criterion $criterion --expl_method Xnesyl-Linear --network XNES-classifier-logistic --input_dataset $dataset --sub_dataset $sub_dataset

            echo "$dataset $sub_dataset CLIP-Linear CLIP-Linear-sample"
            python compute_criterion.py --criterion $criterion --expl_method CLIP-Linear-sample --network CLIP-Linear --input_dataset $dataset --sub_dataset $sub_dataset

        done
    done
done