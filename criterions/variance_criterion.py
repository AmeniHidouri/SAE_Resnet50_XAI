import numpy as np
from PIL import Image

import utils


class VarianceCriterion:
    """Class that implements Variance criterion."""

    def __init__(self, model, perturbation, xai_method, device, xai_id, dataset_name):
        self.model = model
        self.xai_method = xai_method
        self.xai_id_expl = xai_id
        self.perturbation = perturbation
        self.dataset_name = dataset_name
        self.device = device
        self.perturbator = utils.ImagePerturbator()

    def compute_criterion(self, image, len_set=30):
        # Compute the set of perturbated explanations
        pil_image = Image.fromarray(image.cpu().detach().numpy())
        set_perturbated_activations = []
        for magnitude in np.linspace(0.1, 3, len_set):
            perturbated_image = self.perturbator.perturbate_image(
                pil_image, self.perturbation, magnitude
            )
            activation = self.xai_method.compute_and_plot_explanation(
                perturbated_image,
                model=self.model,
                save_expl=False,
                save_activations=False,
                return_activations=True,
            )

            if self.xai_id_expl in ["39", "40", "41", "42", "43", "44", "45", "48", "49"]:
                activation = utils.activations_from_dict(
                    activation, self.dataset_name, self.xai_id_expl
                )
            magintude_explanation = np.sum(abs(activation))
            set_perturbated_activations.append(magintude_explanation)

        return (
            sum([parturbations**2 for parturbations in set_perturbated_activations])
            / len(set_perturbated_activations)
            - sum(set_perturbated_activations) ** 2 / len(set_perturbated_activations) ** 2
        )
