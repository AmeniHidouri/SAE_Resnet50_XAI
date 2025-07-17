import json
from pathlib import Path

import evaluate
import numpy as np
import scipy.stats

import utils


class SimCBM:
    """Class that implements SimCBM criterion."""

    def __init__(self, variant):
        self.variant = variant
        if variant == "bleu":
            self.eval_module = evaluate.load("bleu")

        elif variant == "rouge":
            self.eval_module = evaluate.load("rouge")

    def compute_criterion(self, a_pth, ref_sentence):
        with Path(a_pth).open("r") as fp:
            activations_dict = json.load(fp)

        dataset_name = a_pth.split("/")[2].split("_")[0]
        label = a_pth.split("/")[2].split("_")[-1].split(".")[0]
        xai_id = a_pth.split("/")[-1].split("_")[2]
        activations = utils.activations_from_dict(activations_dict, dataset_name, xai_id)

        if self.variant == "bleu":
            text = utils.sentence_from_activations(activations, dataset_name, xai_id, top_N=5)
            explanation_text = [
                f"The model predicted the image as {label} because it detected the concepts: {text}"
            ]
            return self.eval_module.compute(predictions=explanation_text, references=ref_sentence)[
                "bleu"
            ]

        if self.variant == "rouge":
            text = utils.sentence_from_activations(activations, dataset_name, xai_id, top_N=5)
            explanation_text = [
                f"The model predicted the image as {label} because it detected the concepts: {text}"
            ]
            return self.eval_module.compute(predictions=explanation_text, references=ref_sentence)[
                "rouge1"
            ]

        if self.variant == "entropy":
            # Flatten the saliency map to a 1D array
            flattened_map = activations.flatten()

            # Remove zero values to avoid log(0) issues (which would result in NaN)
            flattened_map = flattened_map[flattened_map > 0]

            # Normalize the saliency map to get probabilities (sum should be 1)
            prob_map = flattened_map / np.sum(flattened_map)

            # Compute entropy using the formula -sum(p(x) * log(p(x)))
            return scipy.stats.entropy(prob_map)

        else:
            ValueError("Invalid variant.")
            return None
