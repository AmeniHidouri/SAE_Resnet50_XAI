import argparse

import numpy as np
import torch

import data
import models
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="pascalpart",
        help="dataset in (pascalpart,monumai,coco,catsdogscars)",
    )
    parser.add_argument("--network", type=str, default="CBM-backbone-resnet")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument(
        "--type_output", type=str, default="float", help="Output type in (bool,float)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Args
    args = parse_args()

    ## Others parameters
    device = f"cuda:{args.gpus}" if (torch.cuda.is_available() and args.gpus != "-1") else "cpu"

    ## Import metadata
    metadata = data.metadata_importer(args.dataset_name)

    ## Import model
    model = models.model_importer(args.network, metadata, device, load_model=True)

    ## Import dataset
    # Load the dataloaders for the training, validation, and test sets
    dataset_train, dataset_val, dataset_test = data.full_dataloader_importer(
        args.dataset_name,
        "images",
        "concepts",
        device,
        training_method="pytorch",
        shuffle=False,
        bs_train=1,
    )

    ## Generate data

    L_splits = ["train", "val", "test"]
    for id_split, dataset in enumerate([dataset_train, dataset_val, dataset_test]):
        print("Split : ", L_splits[id_split])
        score = utils.evaluate_(model, dataset)
        print("Score:", score)
        model.eval()
        if args.type_output == "bool":
            D = {}
            for i, item in enumerate(dataset):
                with torch.no_grad():
                    preds = model(item["input"]).cpu().detach().numpy() > 0
                L = []
                for j, concept in enumerate(metadata["labeled_concepts"]):
                    if preds[0][j]:
                        L.append(concept)
                D[str(i)] = L
            utils.save_as_json(
                D,
                "data_npy/annotation/{}/infer_backbone_{}_{}.json".format(
                    args.dataset_name, args.network.replace("-", "_"), L_splits[id_split]
                ),
            )

        elif args.type_output == "float":
            A = np.zeros((len(dataset), len(metadata["labeled_concepts"])))
            for i, item in enumerate(dataset):
                with torch.no_grad():
                    preds = model(item["input"]).cpu().detach().numpy()
                A[i, :] = preds[0]
            utils.save_as_npy(
                A,
                "data_npy/annotation/{}/infer_backbone_{}_{}.npy".format(
                    args.dataset_name, args.network.replace("-", "_"), L_splits[id_split]
                ),
            )
