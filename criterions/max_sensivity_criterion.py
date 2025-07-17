import numpy as np
import quantus
from PIL import Image


class MaxSensitivityCriterion:
    """Class that implements Max Sensitivity criterion."""

    def __init__(self, model, device, metadata_dataset, expl_method):
        self.model = model.eval()
        self.device = device
        self.metadata = metadata_dataset
        self.criterion = quantus.MaxSensitivity(
            nr_samples=10,
            lower_bound=0.2,
            norm_numerator=quantus.norm_func.fro_norm,
            norm_denominator=quantus.norm_func.fro_norm,
            perturb_func=quantus.perturb_func.uniform_noise,
            similarity_func=quantus.similarity_func.difference,
        )

        def infer_expl(model, inputs, targets, **kwargs):
            # Convert to PIL image
            image = Image.fromarray(np.uint8(inputs[0]).transpose(1, 2, 0))
            expl = expl_method.compute_and_plot_explanation(
                image,
                model=self.model,
                save_expl=False,
                save_activations=False,
                return_activations=True,
            )
            return expl[np.newaxis, :]

        self.infer_expl_fct = infer_expl

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
            explain_func=self.infer_expl_fct,
        )

        return result[0]
