for id_question in Q2 Q3 Q4 Q5 Q6 
do
    for dataset in all
    do
        echo "$dataset PASTA-ridge metrics all "
        python train_pasta.py --network PASTA-ridge --dataset_name true_sklearn --sub_dataset_name all --input_data metrics --type_expl all --alpha 1  --id_question $id_question

        echo "$dataset PASTA-ridge CLIP_CBM_text+CLIP_image_blur all "
        python train_pasta.py --network PASTA-ridge --dataset_name true_sklearn --sub_dataset_name all --input_data CLIP_CBM_text+CLIP_image_blur --type_expl all --alpha 100 --id_question $id_question


        echo "$dataset PASTA-lasso metrics all "
        python train_pasta.py --network PASTA-lasso --dataset_name true_sklearn --sub_dataset_name all --input_data metrics --type_expl all --alpha 0.00005  --id_question $id_question

        echo "$dataset PASTA-lasso CLIP_CBM_text+CLIP_image_blur all "
        python train_pasta.py --network PASTA-lasso --dataset_name true_sklearn --sub_dataset_name all --input_data CLIP_CBM_text+CLIP_image_blur --type_expl all --alpha 0.005  --id_question $id_question


        echo "$dataset PASTA-mlp metrics all "
        python train_pasta.py --network PASTA-mlp --dataset_name true_sklearn --sub_dataset_name all --input_data metrics --type_expl all --alpha 1  --id_question $id_question

        echo "$dataset PASTA-mlp CLIP_CBM_text+CLIP_image_blur all "
        python train_pasta.py --network PASTA-mlp --dataset_name true_sklearn --sub_dataset_name all --input_data CLIP_CBM_text+CLIP_image_blur --type_expl all --alpha 1  --id_question $id_question


        echo "$dataset PASTA-svm metrics all "
        python train_pasta.py --network PASTA-svm --dataset_name true_sklearn --sub_dataset_name all --input_data metrics --type_expl all --alpha 1  --id_question $id_question

        echo "$dataset PASTA-svm CLIP_CBM_text+CLIP_image_blur all "
        python train_pasta.py --network PASTA-svm --dataset_name true_sklearn --sub_dataset_name all --input_data CLIP_CBM_text+CLIP_image_blur --type_expl all --alpha 1  --id_question $id_question

        echo "$dataset PASTA-pytorch metrics all "
        python train_pasta.py --network PASTA-pytorch --dataset_name true_sklearn --sub_dataset_name all --input_data metrics --type_expl all --alpha 1  --id_question $id_question

        echo "$dataset PASTA-pytorch CLIP_CBM_text+CLIP_image_blur all "
        python train_pasta.py --network PASTA-pytorch --dataset_name true_sklearn --sub_dataset_name all --input_data CLIP_CBM_text+CLIP_image_blur --type_expl all --alpha 1  --id_question $id_question
    done
done