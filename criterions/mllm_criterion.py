import numpy as np
from PIL import Image

import utils
from llava_test import LLavaInference


class MLLMCriterion:
    """Class that implements Multimodal Large Language Model criterion."""

    def __init__(self, model, xai_method, mode="saliency"):
        self.model = model
        self.xai_method = xai_method
        self.mode = mode
        self.inference_model = LLavaInference()

    def compute_criterion(self, image, label, activation):
        pil_image = Image.fromarray(image.squeeze(0).cpu().detach().numpy())
        if self.mode == "saliency":  # !!! Only for saliency anc cats_dogs !!!
            activation = self.xai_method.compute_and_plot_explanation(
                pil_image,
                model=self.model,
                save_expl=False,
                save_activations=False,
                return_activations=True,
            )
            array_image = image.squeeze(0).cpu().detach().numpy().transpose(2, 0, 1)
            masked_image = (activation * array_image).astype(np.uint8)
            text_question = utils.template_caption(label)
            pil_masked_image = Image.fromarray(masked_image.transpose(1, 2, 0))
            caption_image = self.inference_model.caption_image(pil_masked_image, text_question)
            print("Caption", caption_image)
            preprompt = utils.preprompt_rating("short_cats_dogs_cars")
            text_description = utils.template_rating(label, caption_image)
            """text_description = 'image features a cat' """
            rating = self.inference_model.answer_text(preprompt + text_description)
            print("Rating", rating)

            if "1" in rating:
                return 1
            if "2" in rating:
                return 2
            if "3" in rating:
                return 3
            if "4" in rating:
                return 4
            if "5" in rating:
                return 5

            ValueError("Rating not found")
            return None
        return None
