#!/bin/bash

for dataset in monumai
do
    for network in resnet50 vitB CLIP-zero-shot
    do 
        for expl_method in LIME SHAP GradCAM AblationCAM EigenCAM EigenGradCAM FullGrad GradCAMPlusPlus GradCAMElementWise HiResCAM ScoreCAM XGradCAM DeepFeatureFactorization
        do
            echo "$dataset $network $expl_method"
            python make_samples.py --network $network --expl_method $expl_method --input_dataset $dataset 
        done
    done
    for expl_method in CLIP-QDA-sample LIME_CBM SHAP_CBM
    do
        echo "$dataset CLIP-QDA $expl_method"
        python make_samples.py --network CLIP-QDA --expl_method $expl_method --input_dataset $dataset 
    done

    for expl_method in LIME_CBM SHAP_CBM
    do
        echo "$dataset CBM-classifier-logistic $expl_method"
        python make_samples.py --network CBM-classifier-logistic --expl_method $expl_method --input_dataset $dataset 
    done

    echo "$dataset XNES-classifier-logistic Xnesyl-Linear"
    python make_samples.py --network XNES-classifier-logistic --expl_method Xnesyl-Linear --input_dataset $dataset 

    echo "$dataset CLIP-Linear CLIP-Linear-sample"
    python make_samples.py --network CLIP-Linear --expl_method CLIP-Linear-sample --input_dataset $dataset 
    
done