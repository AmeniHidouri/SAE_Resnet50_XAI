import numpy as np
import torch
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import data
import models


class ClassificationCriterion:
    """Class that implements Classification criterion."""

    def __init__(self, metadata_dataset, device, classifier):
        self.metadata = metadata_dataset
        self.classifier = classifier

        if "CLIP" in classifier:
            self.model = models.model_importer(
                classifier, metadata_dataset, device, load_model=False
            )

        if classifier == "logistic":
            self.model = LogisticRegression()

        elif classifier == "qda":
            self.model = QuadraticDiscriminantAnalysis()

        elif classifier == "svm":
            self.model = SVC()

    def compute_criterion(self, xai_id):
        activation_info = {"root": "result_dataset", "id_expl": xai_id}
        data_activations = data.sample_dataloader_importer(
            self.metadata["name"], test_mode=False, import_activations=activation_info
        )
        all_dataset_activations = [
            torch.tensor(data_["activations"][0]).detach().cpu().numpy()
            for data_ in data_activations
        ]
        all_dataset_labels = [
            self.metadata["labels"].index(data_["label"][0]) for data_ in data_activations
        ]

        # Shuffle data
        """data_list = list(zip(all_dataset_activations, all_dataset_labels))
        random.seed(493)
        random.shuffle(data_list)
        all_dataset_activations, all_dataset_labels = zip(*data_list)"""

        data_activation = np.array(all_dataset_activations)
        data_labels = np.array(all_dataset_labels)

        # Make train and test set, half train and half test
        data_activation_train = data_activation[: len(data_activation) // 2]
        data_activation_test = data_activation[len(data_activation) // 2 :]
        data_labels_train = data_labels[: len(data_labels) // 2]
        data_labels_test = data_labels[len(data_labels) // 2 :]

        # Perform training
        if "CLIP" in self.classifier:
            self.model.train(data_activation_train, data_labels_train)
            return self.model.test(data_activation_test, data_labels_test)

        self.model.fit(data_activation_train, data_labels_train)
        return self.model.score(data_activation_test, data_labels_test)

        # Compute accuracy
