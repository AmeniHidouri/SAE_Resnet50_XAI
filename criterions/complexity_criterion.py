import numpy as np
import torch
from sklearn.decomposition import PCA

import data


class ComplexityCriterion:
    """Class that implements Complexity criterion."""

    def __init__(self, percent, metadata_dataset):
        self.percent = percent
        self.metadata = metadata_dataset

    def compute_criterion(self, xai_id):
        """Compute the variance ratio of the PCA of a distribution given a list of samples.

        Parameters:


        Returns:
        - A tuple containing:
            - variance_ratio: The explained variance ratio of the selected components.
            - n_components: The number of components required to reach the given percentage of total variance.
        """
        activation_info = {"root": "result_dataset", "id_expl": xai_id}
        data_activations = data.sample_dataloader_importer(
            self.metadata["name"], test_mode=False, import_activations=activation_info
        )
        all_dataset_activations = [
            torch.flatten(data_["activations"]).detach().cpu().numpy() for data_ in data_activations
        ]

        # Convert dataset to numpy array for PCA
        data_activation = np.array(all_dataset_activations)
        # Perform PCA
        pca = PCA()
        pca.fit(data_activation)

        # Compute cumulative explained variance ratio
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

        # Get the explained variance ratio of the selected components
        return cumulative_variance_ratio[
            int((self.percent / 100.0) * len(cumulative_variance_ratio))
        ]
