for dataset in all
do
    for network in PASTA-ridge PASTA-lasso PASTA-svm
    do 
        for input_data in CLIP_image_blur CLIP_heatmap
        do 
            echo "$dataset $network $input_data"
            python train_pasta.py --network $network --sub_dataset_name $dataset --input_data $input_data --type_expl saliency
        done
    done
done