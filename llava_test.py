import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from LLaVA.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from LLaVA.llava.conversation import SeparatorStyle, conv_templates
from LLaVA.llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
from LLaVA.llava.model import LlavaLlamaForCausalLM
from LLaVA.llava.utils import disable_torch_init


class LLavaInference:
    def __init__(self):
        super().__init__()

        model_path = "4bit/llava-v1.5-13b-3GB"
        kwargs = {"device_map": "auto"}
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        ## For download pth on jzay
        """from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
        CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")"""

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = LlavaLlamaForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device="cuda")
        self.image_processor = vision_tower.image_processor

    def caption_image(self, image, prompt):
        disable_torch_init()
        conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        image_tensor = (
            self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            .half()
            .cuda()
        )
        inp = f"{roles[0]}: {prompt}"
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        raw_prompt = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(
                raw_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
            outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
            conv.messages[-1][-1] = outputs
            return outputs.rsplit("</s>", 1)[0]

    def answer_text(self, prompt):
        disable_torch_init()
        conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        raw_prompt = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(
                raw_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        conv.messages[-1][-1] = outputs
        return outputs.rsplit("</s>", 1)[0]


if __name__ == "__main__":
    inference_module = LLavaInference()

    output = inference_module.answer_text("""
        From an image caption, rate as a note from 1 to 5, how the description is good. Here is some examples.
        
        dog
        This explanation focuses on a face-like area with features such as a nose and eyes and some hair on either side of the cheeks, more consistent with a dog. 
        3

        cat
        This explanation focuses on the whole-body part of the animal, in which we can observe its hair and body shape, which is more in line with the characteristics of cats
        4

        cat
        The features in the image indicate that this is a cat.
        1

        car
        This interpretation focuses on the overall contours of the object and features wheels, making it very much in line with being a car. The overall silhouette is also more evidence that it is a car.
        4

        cat
        The image shows a cat lying on a chair, with its eyes glowing red. The cat is positioned on a red cushion, which adds a contrasting element to the scene. The cat's eyes are illuminated by a light source, making them appear red. This effect can be achieved by using a flash or a light source to create a reflection on the cat's eyes. The cat's position on the chair and its glowing eyes are what make it identifiable as a cat.
        4
   
        cat     
        This interpretation focuses on the cat's face, whose pointed ears and furry features indicate that it is most likely a cat
        3

        dog
        The image features a large brown dog standing in a room, with its head turned to the side. The dog appears to be looking at something or someone, possibly its owner. The dog's posture and size suggest that it is a canine, and its presence in the room indicates that it is a domesticated pet. The dog's position and the context of the image make it clear that it is a dog, and not a cat or another animal.
        4

        dog
        This picture is clearly an animal and fits the profile of a dog
        2
        
        Rate this explanation:
        
        car
        This is a dog 
        """)

    print(output)
