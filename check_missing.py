import json
import os
import random
import re
from pathlib import Path

import torch
from PIL import Image

import data
import explanations
import models
import utils
from compute_explanation import make_plots_explanation


def load_json(json_file_path):
    with open(json_file_path) as file:
        return json.load(file)


def extract_ids_from_path(file_path):
    match = re.search(r"_([0-9]+)_([0-9]+)_", file_path)
    if match:
        id_img = int(match.group(1))
        id_xai = int(match.group(2))
        return id_img, id_xai
    return None, None


def get_all_pairs_in_json(data):
    pairs_in_json = set()
    print(len(data))
    for entry in data:
        for key in ["explication"]:
            if "imgs_perturbated_same" in entry["imgpertubated"]:
                file_path = entry[key]
                id_img, id_xai = extract_ids_from_path(file_path)
                if id_img is not None and id_xai is not None:
                    pairs_in_json.add((id_img, id_xai))
    return pairs_in_json


def find_missing_pairs(pairs_in_json, id_img_range, xai_id_range):
    all_possible_pairs = {(id_img, xai_id) for id_img in id_img_range for xai_id in xai_id_range}
    return all_possible_pairs - pairs_in_json

def search_for_other_files(dataset_name, predictedclass, root="result_dataset/imgs/"):
    for root, _, files in os.walk(root):
        # shuffle files
        random.shuffle(files)
        for file in files:
            """if ('_'+predictedclass) not in file and dataset_name in file:
                return root + file"""
            if ("_" + predictedclass) not in file:
                return root + file
    return None


def main(json_file_path):
    id_img_range = range(75,100)  # 0 to 99
    xai_id_range = range(49,50)  # 0 to 49

    data = load_json(json_file_path)
    pairs_in_json = get_all_pairs_in_json(data)
    missing_pairs = find_missing_pairs(pairs_in_json, id_img_range, xai_id_range)

    print(f"Missing pairs of (id_img, id_xai): {len(missing_pairs)}")

    for pair in sorted(missing_pairs):
        print(f"- {pair}")
        compute_missing_sample_same(pair[1],pair[0])
        '''compute_missing_sample_diff(pair[1], pair[0])'''

    exit()

def find_img_path(img_id, root="result_dataset/imgs/"):
    for root, _, files in os.walk(root):
        for file in files:
            if "_" + str(img_id) + "_" in file:
                return root + file

    return None


def compute_missing_sample_diff(xai_id, img_id):
    with Path("result_dataset/expl.json").open("r") as fp:
        listobj = json.load(fp)

    # Load sample
    input_image_name = find_img_path(img_id)
    GTclass = re.search(r"\d+_(.+)\.png", input_image_name).group(1)
    id_sample = str(img_id)
    input_image = Image.open(input_image_name).convert("RGB")
    Datasetname = input_image_name.split("/")[-1].split("_")[0]
    XAI_id = xai_id
    image_perturbator = utils.ImagePerturbator()

    end_activations_path = "json" if int(XAI_id) > 38 else "npy"

    # Load data/explainer/model
    device = "cuda:0" if (torch.cuda.is_available()) else "cpu"

    metadata = data.metadata_importer(Datasetname)

    # Name method
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
        "CLIP-Linear-sample",
        "LIME_CBM (CLIP-QDA)",
        "SHAP_CBM (CLIP-QDA)",
        "LIME_CBM (CBM-classifier-logistic)",
        "SHAP_CBM (CBM-classifier-logistic)",
        "Xnesyl-Linear",
        "BCos (resnet50-bcos)",
        "BCos (vitB-bcos)",
        "Rise_CBM (CLIP-QDA)",
        "Rise_CBM (CBM-classifier-logistic)",
    ]

    name_method = list_xai_methods[int(XAI_id)]

    if name_method == "CLIP-QDA-sample":
        network_name = "CLIP-QDA"

    elif name_method == "CLIP-Linear-sample":
        network_name = "CLIP-Linear"

    elif name_method == "Xnesyl-Linear":
        network_name = "XNES-classifier-logistic"

    else:
        network_name = name_method.split("(")[1].split(")")[0]

    model = models.model_importer(network_name, metadata, device, load_model=True)
    explanation_method = name_method.split("(")[0].replace(" ", "")

    # Compute explanations
    explanation_method = explanations.explanation_method_importer(
        explanation_method,
        device=device,
        model=model,
        model_name=network_name,
        metadata=metadata,
    )

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

    pth_other_img = search_for_other_files(Datasetname, predictedclass)

    print("PATH", pth_other_img, input_image_name)

    other_img = Image.open(pth_other_img).convert("RGB")

    same = True  # Flag to track if the perturbated image has the same classification as the original one

    # Search for an image from the same dataset that have a different class

    magnitude = 0.1

    while same:
        if magnitude >= 1:
            print("No label-changing perturbation found!")
            break

        ## Save, if feasible, a description of a modified image that results in a diff classification as the original one

        # Define paths of the perturbated image with the classification filled
        path_expl_perturb_diff_class = f"result_dataset/expl_perturbated_diff/{Datasetname}_{id_sample}_{13}_{XAI_id}_FILL_CLASS.png"
        path_img_perturb_diff_class = f"result_dataset/imgs_perturbated_diff/{Datasetname}_{id_sample}_{13}_{XAI_id}_FILL_CLASS.png"
        path_activations_perturb_diff_class = f"result_dataset/activations_perturbated_diff/{Datasetname}_{id_sample}_{13}_{XAI_id}_FILL_CLASS.{end_activations_path}"

        # Perturbate the image
        img_perturb = image_perturbator.mixup_perturbation(
            input_image, other_img, magnitude=magnitude
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

        # If the perturbated image has not the same classification, save the information
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

        # If the perturbated image has a different classification, remove the files
        else:
            Path(path_expl_perturb_class).unlink()
            Path(path_img_perturb_class).unlink()
            Path(path_activations_perturb_class).unlink()

        # Increment the perturbation counter and decrease the magnitude
        magnitude += 0.1

    with Path("result_dataset/expl.json").open("w") as json_file:
        json.dump(listobj, json_file, indent=4, separators=(",", ": "))

    print("Done!", path_expl_perturb_class)

def compute_missing_sample_same(xai_id, img_id):
    with Path("result_dataset/expl.json").open("r") as fp:
        listobj = json.load(fp)

    # Load sample
    input_image_name = find_img_path(img_id)
    GTclass = re.search(r"\d+_(.+)\.png", input_image_name).group(1)
    id_sample = str(img_id)
    input_image = Image.open(input_image_name).convert("RGB")
    Datasetname = input_image_name.split("/")[-1].split("_")[0]
    XAI_id = xai_id
    image_perturbator = utils.ImagePerturbator()

    end_activations_path = "json" if int(XAI_id) > 38 else "npy"

    # Load data/explainer/model
    device = "cuda:0" if (torch.cuda.is_available()) else "cpu"

    metadata = data.metadata_importer(Datasetname)

    # Name method
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
        "CLIP-Linear-sample",
        "LIME_CBM (CLIP-QDA)",
        "SHAP_CBM (CLIP-QDA)",
        "LIME_CBM (CBM-classifier-logistic)",
        "SHAP_CBM (CBM-classifier-logistic)",
        "Xnesyl-Linear",
        "BCos (resnet50-bcos)",
        "BCos (vitB-bcos)",
        "Rise_CBM (CLIP-QDA)",
        "Rise_CBM (CBM-classifier-logistic)",
    ]

    name_method = list_xai_methods[int(XAI_id)]

    if name_method == "CLIP-QDA-sample":
        network_name = "CLIP-QDA"

    elif name_method == "CLIP-Linear-sample":
        network_name = "CLIP-Linear"

    elif name_method == "Xnesyl-Linear":
        network_name = "XNES-classifier-logistic"

    else:
        network_name = name_method.split("(")[1].split(")")[0]

    model = models.model_importer(network_name, metadata, device, load_model=True)
    explanation_method = name_method.split("(")[0].replace(" ", "")

    # Compute explanations
    explanation_method = explanations.explanation_method_importer(
        explanation_method,
        device=device,
        model=model,
        model_name=network_name,
        metadata=metadata,
    )

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

    N_perturbs = len(image_perturbator.perturbation_methods)
    l_id_perturb = list(range(N_perturbs))

    same = False  # Flag to track if the perturbated image has the same classification as the original one
    count_magn_perturb = 0  # Counter of magnutude of perturbations tried

    while not same:
        if count_magn_perturb == 0:
            id_perturb = random.choice(l_id_perturb)
            l_id_perturb.pop(l_id_perturb.index(id_perturb))
            count_magn_perturb = 0  # Counter of magnutude of perturbations tried
            magnitude = 1  # Magnitude of perturbations

            if l_id_perturb == []:
                print("No label-keeping perturbation found!")
                break

        ## Save, if feasible, a description of a modified image that results in a same classification as the original one

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
        count_magn_perturb += 1
        magnitude *= 0.9

        if count_magn_perturb == 11:
            count_magn_perturb = 0

    with Path("result_dataset/expl.json").open("w") as json_file:
        json.dump(listobj, json_file, indent=4, separators=(",", ": "))

    print("Done!", path_expl_perturb_class)

if __name__ == "__main__":
    json_file_path = "result_dataset/expl.json"
    main(json_file_path)


