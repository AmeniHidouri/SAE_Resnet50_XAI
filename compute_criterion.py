import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

import criterions
import data
import explanations
import models
import utils

if __name__ == "__main__":
    ## Parser
    def parse_args():
        # Training settings
        parser = argparse.ArgumentParser(description="")
        parser.add_argument(
            "--network",
            type=str,
            default="resnet50",
            help="network (resnet50, vitB, CLIP-QDA, CLIP-zero-shot, CLIP-Linear, CBM-classifier-logistic, CLIP-LaBo, XNES-classifier-logistic)",
        )
        parser.add_argument("--gpus", type=str, default="0", help="gpu ids")
        parser.add_argument(
            "--expl_method",
            type=str,
            default="GradCAM",
            help="explanation method (Rise_CBM, LIME, SHAP, GradCAM, CLIP-QDA-sample, LIME_CBM, SHAP_CBM, AblationCAM, EigenCAM, EigenGradCAM, FullGrad, GradCAMPlusPlus, GradCAMElementWise, HiResCAM, ScoreCAM, XGradCAM, DeepFeatureFactorization, CLIP-Linear-sample, CLIP-LaBo-sample, Xnesyl-Linear)",
        )
        parser.add_argument(
            "--input_dataset",
            type=str,
            default="toy",
            help="set of input data to use in [toy,true]",
        )
        parser.add_argument(
            "--criterion",
            type=str,
            default="variance_gauss",
            help="set of input data to use in [mllm,variance_gauss,variance_sharp,variance_bright,complexity_10,complexity_20,complexity_30,classification_logistic,classification_qda,classification_svm,faithfullness,maxsensitivity]",
        )
        parser.add_argument(
            "--sub_dataset",
            type=str,
            default="coco",
            help="dataset of interest in [pascalpart,monumai,coco,catsdogscars]",
        )
        return parser.parse_args()

    args = parse_args()

    # Load dataset
    dataset = data.pasta_dataloader_importer(
        args.input_dataset, import_all=True, sub_dataset=args.sub_dataset
    )
    metadata = data.metadata_importer(args.sub_dataset)

    # Load explainer/model
    args = parse_args()
    device = f"cuda:{args.gpus}" if (torch.cuda.is_available() and args.gpus != "-1") else "cpu"
    model = models.model_importer(args.network, metadata, device, load_model=True, bcos_eval=True)
    explanation_method = explanations.explanation_method_importer(
        args.expl_method,
        device=device,
        model=model,
        model_name=args.network,
        metadata=metadata,
    )

    xai_id_expl = str(utils.xai_id_from_model_expl(args.expl_method, args.network))

    with Path(f"result_dataset/criterion_{args.input_dataset}.json").open("r") as fp:
        listobj = json.load(fp)

    # Load criterion
    if xai_id_expl in ["39", "40", "41", "42", "43", "44", "45", "48", "49"]:
        criterion = criterions.criterion_importer(
            args.criterion,
            model,
            metadata,
            explanation_method,
            device,
            type_expl="cbm",
            xai_id=xai_id_expl,
            dataset_name=args.sub_dataset,
        )

    else:
        criterion = criterions.criterion_importer(
            args.criterion, model, metadata, explanation_method, device, type_expl="saliency"
        )

    # Compute criterion

    if "variance" in args.criterion:
        for data_ in tqdm(dataset):
            img_id = data_["img_id"][0]
            xai_id = data_["xai_id"][0]
            if xai_id_expl == str(xai_id):
                criterion_value = criterion.compute_criterion(data_["image"][0])
                if f"{img_id}_{xai_id}" not in listobj:
                    listobj[f"{img_id}_{xai_id}"] = {}
                # If the entry is not already in listobj, create it
                if f"{img_id}_{xai_id}" not in listobj:
                    listobj[f"{img_id}_{xai_id}"] = {}
                listobj[f"{img_id}_{xai_id}"][args.criterion] = criterion_value

    elif True:
        for data_ in tqdm(dataset):
            img_id = data_["img_id"][0]
            xai_id = data_["xai_id"][0]
            if xai_id_expl == str(xai_id):
                criterion_value = criterion.compute_criterion(
                    data_["image"][0], data_["GT_class"], data_["activations"][0]
                )
                if f"{img_id}_{xai_id}" not in listobj:
                    listobj[f"{img_id}_{xai_id}"] = {}
                # If the entry is not already in listobj, create it
                if f"{img_id}_{xai_id}" not in listobj:
                    listobj[f"{img_id}_{xai_id}"] = {}
                listobj[f"{img_id}_{xai_id}"][args.criterion] = criterion_value

    elif True:
        xai_id = utils.xai_id_from_model_expl(args.expl_method, args.network)
        criterion_value = criterion.compute_criterion(xai_id)
        for data_ in tqdm(dataset):
            img_id = data_["img_id"][0]
            xai_id = data_["xai_id"][0]
            if xai_id_expl == str(xai_id):
                if f"{img_id}_{xai_id}" not in listobj:
                    listobj[f"{img_id}_{xai_id}"] = {}
                listobj[f"{img_id}_{xai_id}"][args.criterion] = float(criterion_value)

    with Path(f"result_dataset/criterion_{args.input_dataset}.json").open("w") as json_file:
        json.dump(listobj, json_file, indent=4, separators=(",", ": "))
