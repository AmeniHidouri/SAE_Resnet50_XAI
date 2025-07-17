for sigma in 05 1 15
do
    for dataset in all
    do
        echo "$dataset PASTA-ridge metrics saliency $sigma"
        python train_pasta.py --network PASTA-ridge --sub_dataset_name $dataset --input_data metrics --type_expl saliency --alpha 0.001 --sigma_noise $sigma

        for input_data in CLIP_image_blur CLIP_heatmap
        do 
            echo "$dataset PASTA-ridge $input_data saliency $sigma"
            python train_pasta.py --network PASTA-ridge --sub_dataset_name $dataset --input_data $input_data --type_expl saliency --alpha 100 --sigma_noise $sigma
        done

        echo "$dataset PASTA-ridge metrics cbm $sigma"
        python train_pasta.py --network PASTA-ridge --sub_dataset_name $dataset --input_data metrics --type_expl cbm --alpha 0.001 --sigma_noise $sigma

        echo "$dataset PASTA-ridge CBM_activations cbm $sigma"
        python train_pasta.py --network PASTA-ridge --sub_dataset_name $dataset --input_data CBM_activations --type_expl cbm --alpha 100 --sigma_noise $sigma

        echo "$dataset PASTA-ridge CLIP_CBM_text cbm $sigma"
        python train_pasta.py --network PASTA-ridge --sub_dataset_name $dataset --input_data CLIP_CBM_text --type_expl cbm --alpha 1 --sigma_noise $sigma


        echo "$dataset PASTA-lasso metrics saliency $sigma"
        python train_pasta.py --network PASTA-lasso --sub_dataset_name $dataset --input_data metrics --type_expl saliency --alpha 0.00005 --sigma_noise $sigma

        for input_data in CLIP_image_blur CLIP_heatmap
        do 
            echo "$dataset PASTA-lasso $input_data saliency $sigma"
            python train_pasta.py --network PASTA-lasso --sub_dataset_name $dataset --input_data $input_data --type_expl saliency --alpha 0.005 --sigma_noise $sigma
        done

        echo "$dataset PASTA-lasso metrics cbm $sigma"
        python train_pasta.py --network PASTA-lasso --sub_dataset_name $dataset --input_data metrics --type_expl cbm --alpha 0.00005 --sigma_noise $sigma

        echo "$dataset PASTA-lasso CBM_activations cbm $sigma"
        python train_pasta.py --network PASTA-lasso --sub_dataset_name $dataset --input_data CBM_activations --type_expl cbm --alpha 005 --sigma_noise $sigma

        echo "$dataset PASTA-lasso CLIP_CBM_text cbm $sigma"
        python train_pasta.py --network PASTA-lasso --sub_dataset_name $dataset --input_data CLIP_CBM_text --type_expl cbm --sigma_noise $sigma


        echo "$dataset PASTA-svm metrics saliency $sigma"
        python train_pasta.py --network PASTA-svm --sub_dataset_name $dataset --input_data metrics --type_expl saliency --sigma_noise $sigma

        for input_data in CLIP_image_blur CLIP_heatmap
        do 
            echo "$dataset PASTA-svm $input_data saliency $sigma"
            python train_pasta.py --network PASTA-svm --sub_dataset_name $dataset --input_data $input_data --type_expl saliency --sigma_noise $sigma
        done

        echo "$dataset PASTA-svm metrics cbm $sigma"
        python train_pasta.py --network PASTA-svm --sub_dataset_name $dataset --input_data metrics --type_expl cbm --sigma_noise $sigma

        echo "$dataset PASTA-svm CBM_activations cbm $sigma"
        python train_pasta.py --network PASTA-svm --sub_dataset_name $dataset --input_data CBM_activations --type_expl cbm --sigma_noise $sigma

        echo "$dataset PASTA-svm CLIP_CBM_text cbm $sigma"
        python train_pasta.py --network PASTA-svm --sub_dataset_name $dataset --input_data CLIP_CBM_text --type_expl cbm --sigma_noise $sigma
    
        echo "$dataset PASTA-mlp metrics saliency $sigma"
        python train_pasta.py --network PASTA-mlp --sub_dataset_name $dataset --input_data metrics --type_expl saliency --sigma_noise $sigma

        for input_data in CLIP_image_blur CLIP_heatmap
        do 
            echo "$dataset PASTA-mlp $input_data saliency $sigma"
            python train_pasta.py --network PASTA-mlp --sub_dataset_name $dataset --input_data $input_data --type_expl saliency --sigma_noise $sigma
        done

        echo "$dataset PASTA-mlp metrics cbm $sigma"
        python train_pasta.py --network PASTA-mlp --sub_dataset_name $dataset --input_data metrics --type_expl cbm --sigma_noise $sigma

        echo "$dataset PASTA-mlp CBM_activations cbm $sigma"
        python train_pasta.py --network PASTA-mlp --sub_dataset_name $dataset --input_data CBM_activations --type_expl cbm --sigma_noise $sigma

        echo "$dataset PASTA-mlp CLIP_CBM_text cbm $sigma"
        python train_pasta.py --network PASTA-mlp --sub_dataset_name $dataset --input_data CLIP_CBM_text --type_expl cbm --sigma_noise $sigma

        echo "$dataset PASTA-ridge metrics all $sigma"
        python train_pasta.py --network PASTA-ridge --sub_dataset_name all --input_data metrics --type_expl all --alpha 1 --sigma $sigma

        echo "$dataset PASTA-ridge metrics all $sigma"
        python train_pasta.py --network PASTA-ridge --sub_dataset_name all --input_data CLIP_CBM_text+CLIP_image_blur --type_expl all --alpha 100 --sigma $sigma


        echo "$dataset PASTA-lasso metrics all $sigma"
        python train_pasta.py --network PASTA-lasso --sub_dataset_name all --input_data metrics --type_expl all --alpha 0.00005 --sigma $sigma

        echo "$dataset PASTA-lasso metrics all $sigma"
        python train_pasta.py --network PASTA-lasso --sub_dataset_name all --input_data CLIP_CBM_text+CLIP_image_blur --type_expl all --alpha 0.005 --sigma $sigma


        echo "$dataset PASTA-mlp metrics all $sigma"
        python train_pasta.py --network PASTA-mlp --sub_dataset_name all --input_data metrics --type_expl all --alpha 1 --sigma $sigma

        echo "$dataset PASTA-mlp metrics all $sigma"
        python train_pasta.py --network PASTA-mlp --sub_dataset_name all --input_data CLIP_CBM_text+CLIP_image_blur --type_expl all --alpha 1 --sigma $sigma


        echo "$dataset PASTA-svm metrics all $sigma"
        python train_pasta.py --network PASTA-ridge --sub_dataset_name all --input_data metrics --type_expl all --alpha 1 --sigma $sigma

        echo "$dataset PASTA-svm metrics all $sigma"
        python train_pasta.py --network PASTA-ridge --sub_dataset_name all --input_data CLIP_CBM_text+CLIP_image_blur --type_expl all --alpha 1 --sigma $sigma


    done
done