import numpy as np
import quantus


class FaithfullnessCriterion:
    """Class that implements Faithfulness criterion."""

    def __init__(self, model, device, metadata_dataset):
        self.model = model.eval()
        self.device = device
        self.metadata = metadata_dataset
        self.criterion = quantus.FaithfulnessCorrelation(
            nr_runs=100,
            subset_size=224,
            perturb_baseline="black",
            perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            similarity_func=quantus.similarity_func.correlation_pearson,
            abs=False,
            return_aggregate=False,
        )

    def compute_criterion(self, image, label, activation):
        # Compute the set of perturbated explanations

        image = image[np.newaxis, :, :, :].permute(0, 3, 1, 2)
        activation = activation[np.newaxis, :, :]
        label = np.array([self.metadata["labels"].index(label[0])])[np.newaxis, :]

        result = self.criterion(
            model=self.model,
            x_batch=image.cpu().numpy(),
            y_batch=label,
            a_batch=activation.cpu().numpy(),
            device=self.device,
        )

        return result[0]
