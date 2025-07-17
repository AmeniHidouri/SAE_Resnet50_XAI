import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import data
import explanations
import models
import utils
from compute_explanation import make_plots_explanation

if __name__ == "__main__":

    def parse_args():
        # Training settings
        parser = argparse.ArgumentParser(description="")
        parser.add_argument(
            "--network",
            type=str,
            default="resnet50",
            help="network (resnet50 or ViTB or CLIP-QDA or CLIP-zero-shot)",
        )
        parser.add_argument("--gpus", type=str, default="0", help="gpu ids")  # TODO multigpu
        parser.add_argument(
            "--expl_method",
            type=str,
            default="explanation method (LIME, SHAP, GradCAM, CLIP-QDA-sample, LIME-CBM, SHAP-CBM, AblationCAM, EigenCAM, EigenGradCAM, FullGrad, GradCAMPlusPlus, GradCAMElementWise, HiResCAM, ScoreCAM, XGradCAM)",
            help="gpu ids",
        )
        parser.add_argument(
            "--input_dataset",
            type=str,
            default="pascalpart",
            help="set of input data to use in [pascalpart,monumai]",
        )
        parser.add_argument(
            "--test_model",
            action="store_true",
            help="run the experiment only on one sample for testing the methods",
        )
        parser.add_argument(
            "--no_perturb",
            action="store_true",
            help="run the experiment but not on perturbed samples",
        )

        return parser.parse_args()

    # Load data/explainer/model
    args = parse_args()
    device = f"cuda:{args.gpus}" if (torch.cuda.is_available() and args.gpus != "-1") else "cpu"

    metadata = data.metadata_importer(args.input_dataset)
    model = models.model_importer(args.network, metadata, device, load_model=True)

    # If explanation as concepts importance, save as json if concepts as saliency map, save as npy
    if args.expl_method in [
        "CLIP-QDA-sample",
        "LIME_CBM",
        "SHAP_CBM",
        "Xnesyl-Linear",
        "CLIP-Linear-sample",
        "CLIP-LaBo-sample",
    ]:
        end_activations_path = "json"

    else:
        end_activations_path = "npy"

    # Compute explanations
    explanation_method = explanations.explanation_method_importer(
        args.expl_method,
        device=device,
        model=model,
        model_name=args.network,
        metadata=metadata,
    )

    # Import dataset samples
    dataset_sample = data.sample_dataloader_importer(args.input_dataset, test_mode=args.test_model)

    # Define explanation
    if (
        args.network == "CLIP-zero-shot"
        or args.network == "resnet50"
        or args.network == "vitB"
        or (args.network == "CLIP-QDA" and args.expl_method != "CLIP-QDA-sample")
        or args.network == "CBM-classifier-logistic"
    ):
        expl_method_full = args.expl_method + f" ({args.network})"

    else:
        expl_method_full = args.expl_method

    # Methods available
    list_xai_methods = [
        "GradCAM (resnet50)",
        "GradCAM (vitB)",
        "GradCAM (CLIP-zero-shot)",
        "LIME (resnet50)",
        "LIME (vitB)",
        "LIME (CLIP-zero-shot)",
        "SHAP (resnet50)",
        "SHAP (vitB)",
        "SHAP (CLIP-zero-shot)",
        "AblationCAM (resnet50)",
        "AblationCAM (vitB)",
        "AblationCAM (CLIP-zero-shot)",
        "EigenCAM (resnet50)",
        "EigenCAM (vitB)",
        "EigenCAM (CLIP-zero-shot)",
        "EigenGradCAM (resnet50)",
        "EigenGradCAM (vitB)",
        "EigenGradCAM (CLIP-zero-shot)",
        "FullGrad (resnet50)",
        "FullGrad (vitB)",
        "FullGrad (CLIP-zero-shot)",
        "GradCAMPlusPlus (resnet50)",
        "GradCAMPlusPlus (vitB)",
        "GradCAMPlusPlus (CLIP-zero-shot)",
        "GradCAMElementWise (resnet50)",
        "GradCAMElementWise (vitB)",
        "GradCAMElementWise (CLIP-zero-shot)",
        "HiResCAM (resnet50)",
        "HiResCAM (vitB)",
        "HiResCAM (CLIP-zero-shot)",
        "ScoreCAM (resnet50)",
        "ScoreCAM (vitB)",
        "ScoreCAM (CLIP-zero-shot)",
        "XGradCAM (resnet50)",
        "XGradCAM (vitB)",
        "XGradCAM (CLIP-zero-shot)",
        "DeepFeatureFactorization (resnet50)",
        "DeepFeatureFactorization (vitB)",
        "DeepFeatureFactorization (CLIP-zero-shot)",
        "CLIP-QDA-sample",
        "LIME_CBM (CLIP-QDA)",
        "SHAP_CBM (CLIP-QDA)",
        "LIME_CBM (CBM-classifier-logistic)",
        "SHAP_CBM (CBM-classifier-logistic)",
        "Xnesyl-Linear",
    ]

    with Path("result_dataset/expl.json").open("r") as fp:
        listobj = json.load(fp)

    for data_sample in dataset_sample:
        # Load sample
        GTclass = data_sample["label"][0]
        id_sample = data_sample["id_sample"][0]
        input_image = data_sample["image"][0]
        input_image = Image.fromarray(np.uint8(input_image)).convert("RGB")
        Datasetname = args.input_dataset
        XAI_id = list_xai_methods.index(expl_method_full)
        image_perturbator = utils.ImagePerturbator()

        path_expl = f"result_dataset/expl/{Datasetname}_{id_sample}_{XAI_id}_{GTclass}.png"
        path_img_resize = f"result_dataset/imgs/{Datasetname}_{id_sample}_{GTclass}.png"
        path_activations = f"result_dataset/activations/{Datasetname}_{id_sample}_{XAI_id}_{GTclass}.{end_activations_path}"

        # Save image+explanation+activations
        predictedid = make_plots_explanation(
            model,
            input_image,
            explanation_method,
            save_img=path_img_resize,
            save_expl=path_expl,
            save_activations=path_activations,
        )
        predictedclass = metadata["labels"][predictedid]

        if args.test_model:
            l_id_perturb = [3]  # Test on perturbation == gaussian blur
            print("Tested method", expl_method_full)
        elif args.no_perturb:
            D_data = {
                "ID": id_sample,  # Sample ID
                "img": path_img_resize,  # Path of the resized image
                "explication": path_expl,  # Path of the explanation
                "activations": path_activations,  # Path of the activations
                "predictedclass": predictedclass,  # Classification of the original image
                "GTclass": GTclass,  # Ground truth class
            }  # Classification of the perturbated image
            listobj.append(D_data)
            continue
        else:
            N_perturbs = len(image_perturbator.perturbation_methods)
            l_id_perturb = range(N_perturbs)

        for id_perturb in tqdm(l_id_perturb):
            ## Save, if feasible, a description of a modified image that results in a classification different from the original one

            same = False  # Flag to track if the perturbated image has the same classification as the original one
            Limit_perturb = 10  # Limit of perturbations to apply
            j = 0  # Counter of perturbations
            magnitude = 1  # Magnitude of perturbations

            # Loop over the perturbations
            while not same and (j < Limit_perturb):
                # Define paths of the perturbated image with the classification filled
                path_expl_perturb_same_class = f"result_dataset/expl_perturbated_same/{Datasetname}_{id_sample}_{id_perturb}_{XAI_id}_FILL_CLASS.png"
                path_img_perturb_same_class = f"result_dataset/imgs_perturbated_same/{Datasetname}_{id_sample}_{id_perturb}_{XAI_id}_FILL_CLASS.png"
                path_activations_perturb_same_class = f"result_dataset/activations_perturbated_same/{Datasetname}_{id_sample}_{id_perturb}_{XAI_id}_FILL_CLASS.{end_activations_path}"

                # Perturbate the image
                img_perturb = image_perturbator.perturbate_image(
                    input_image, magnitude=magnitude, id_perturbation=id_perturb
                )

                # Compute the explanation of the perturbated image
                predictedid_perturb = make_plots_explanation(
                    model,
                    img_perturb,
                    explanation_method,
                    save_img=path_img_perturb_same_class,
                    save_expl=path_expl_perturb_same_class,
                    save_activations=path_activations_perturb_same_class,
                )
                predictedclass_perturb = metadata["labels"][predictedid_perturb]

                # Paths of the perturbated image with the classification filled
                path_expl_perturb_class = path_expl_perturb_same_class.replace(
                    "FILL_CLASS", predictedclass_perturb
                )
                path_img_perturb_class = path_img_perturb_same_class.replace(
                    "FILL_CLASS", predictedclass_perturb
                )
                path_activations_perturb_class = path_activations_perturb_same_class.replace(
                    "FILL_CLASS", predictedclass_perturb
                )

                # Rename the perturbated image files
                Path(path_expl_perturb_same_class).rename(path_expl_perturb_class)
                Path(path_img_perturb_same_class).rename(path_img_perturb_class)
                Path(path_activations_perturb_same_class).rename(path_activations_perturb_class)

                # Check if the perturbated image has the same classification as the original one
                same = predictedclass_perturb == predictedclass

                # If the perturbated image has the same classification, save the information
                if same:
                    D_data = {
                        "ID": id_sample,  # Sample ID
                        "img": path_img_resize,  # Path of the resized image
                        "explication": path_expl,  # Path of the explanation
                        "activations": path_activations,  # Path of the activations
                        "predictedclass": predictedclass,  # Classification of the original image
                        "GTclass": GTclass,  # Ground truth class
                        "imgpertubated": path_img_perturb_class,  # Path of the perturbated image
                        "explicationpertubated": path_expl_perturb_class,  # Path of the explanation of the perturbated image
                        "activationspertubated": path_activations_perturb_class,  # Path of the activations of the perturbated image
                        "predictedclasspertubated": predictedclass_perturb,
                    }  # Classification of the perturbated image
                    listobj.append(D_data)

                # If the perturbated image has a different classification, remove the files
                else:
                    Path(path_expl_perturb_class).unlink()
                    Path(path_img_perturb_class).unlink()
                    Path(path_activations_perturb_class).unlink()

                # Increment the perturbation counter and decrease the magnitude
                j += 1
                magnitude *= 0.9

                # Print a warning if no perturbation found
                if j > Limit_perturb:
                    print("Warning! No perturbation found! (same)")

            ## Save, if feasible, a description of a modified image that results in a classification different from the original one

            same = True  # Flag to track if the perturbated image has the same classification as the original one
            Limit_perturb = 10  # Limit of perturbations to apply
            j = 0  # Counter of perturbations
            magnitude = 1  # Magnitude of perturbations

            while same and (j < Limit_perturb):
                # Define paths of the perturbated image with the classification filled
                path_expl_perturb_diff_class = f"result_dataset/expl_perturbated_diff/{Datasetname}_{id_sample}_{id_perturb}_{XAI_id}_FILL_CLASS.png"
                path_img_perturb_diff_class = f"result_dataset/imgs_perturbated_diff/{Datasetname}_{id_sample}_{id_perturb}_{XAI_id}_FILL_CLASS.png"
                path_activations_perturb_diff_class = f"result_dataset/activations_perturbated_diff/{Datasetname}_{id_sample}_{id_perturb}_{XAI_id}_FILL_CLASS.{end_activations_path}"

                # Perturbate the image
                img_perturb = image_perturbator.perturbate_image(
                    input_image, magnitude=magnitude, id_perturbation=id_perturb
                )

                # Compute the explanation of the perturbated image
                predictedid_perturb = make_plots_explanation(
                    model,
                    img_perturb,
                    explanation_method,
                    save_img=path_img_perturb_diff_class,
                    save_expl=path_expl_perturb_diff_class,
                    save_activations=path_activations_perturb_diff_class,
                )
                predictedclass_perturb = metadata["labels"][predictedid_perturb]

                # Paths of the perturbated image with the classification filled
                path_expl_perturb_class = path_expl_perturb_diff_class.replace(
                    "FILL_CLASS", predictedclass_perturb
                )
                path_img_perturb_class = path_img_perturb_diff_class.replace(
                    "FILL_CLASS", predictedclass_perturb
                )
                path_activations_perturb_class = path_activations_perturb_diff_class.replace(
                    "FILL_CLASS", predictedclass_perturb
                )

                # Rename the perturbated image files
                Path(path_expl_perturb_diff_class).rename(path_expl_perturb_class)
                Path(path_img_perturb_diff_class).rename(path_img_perturb_class)
                Path(path_activations_perturb_diff_class).rename(path_activations_perturb_class)

                # Check if the perturbated image has the same classification as the original one
                same = predictedclass_perturb == predictedclass

                # If the perturbated image has a different classification, save the information
                if not same:
                    D_data = {
                        "ID": id_sample,  # Sample ID
                        "img": path_img_resize,  # Path of the resized image
                        "explication": path_expl,  # Path of the explanation
                        "activations": path_activations,  # Path of the activations
                        "predictedclass": predictedclass,  # Classification of the original image
                        "GTclass": GTclass,  # Ground truth class
                        "imgpertubated": path_img_perturb_class,  # Path of the perturbated image
                        "explicationpertubated": path_expl_perturb_class,  # Path of the explanation of the perturbated image
                        "activationspertubated": path_activations_perturb_class,  # Path of the activations of the perturbated image
                        "predictedclasspertubated": predictedclass_perturb,
                    }  # Classification of the perturbated image

                    listobj.append(D_data)

                # If the perturbated image has the same classification, remove the files
                else:
                    Path(path_expl_perturb_class).unlink()
                    Path(path_img_perturb_class).unlink()
                    Path(path_activations_perturb_class).unlink()

                # Increment the perturbation counter and increase the magnitude
                j += 1
                magnitude += 1

                # Print a warning if no perturbation found
                if j >= Limit_perturb:
                    print("Warning! No perturbation found! (diff)")

    with Path("result_dataset/expl.json").open("w") as json_file:
        json.dump(listobj, json_file, indent=4, separators=(",", ": "))
