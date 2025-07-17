#!/bin/bash

for dataset in toy
do
    for sub_dataset in coco
    do
        for criterion in complexity_10 complexity_20 complexity_30 classification_logistic classification_qda classification_svm variance_gauss variance_sharp variance_bright faithfullness
        do
            for network in resnet50 vitB CLIP-zero-shot
            do 
                for expl_method in LIME SHAP GradCAM AblationCAM EigenCAM EigenGradCAM FullGrad GradCAMPlusPlus GradCAMElementWise HiResCAM ScoreCAM XGradCAM DeepFeatureFactorization
                do
                    echo "$dataset $sub_dataset $criterion $network $expl_method" 
                    python compute_criterion.py --criterion $criterion --expl_method $expl_method --network $network --input_dataset $dataset
                done
            done
        done
    done
done