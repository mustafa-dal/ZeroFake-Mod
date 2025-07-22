from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torch.nn.functional as nnf

from diffusers import DDIMScheduler, StableDiffusionPipeline

path = "/home/c01zesh/CISPA-projects/meta_transfer-2023/stable-diffusion/stable-diffusion-v1-4"
path = Path(path).expanduser()

from PIL import Image
from torchvision import transforms

import inspect

import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
try:
    from diffusers.pipeline_utils import DiffusionPipeline
except:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler,PNDMScheduler, LMSDiscreteScheduler
from diffusers.utils import deprecate, logging

from PIL import Image
import requests
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from blipmodels import blip_decoder # TO LOOK FOR
import spacy

import os
import sys
import argparse

"""
replace_first_noun(..) -> 
Use spacy to obtain linguistic annotations for each word or token in the sentence. Then replaces the first noun found in the prompt.
"""
def replace_first_noun(sentence, replacement):
    doc = nlp(sentence)
    for token in doc:

        if token.pos_ == "NOUN":
            return sentence[:token.idx] + replacement + sentence[token.idx + len(token.text):]

    return sentence

logger = logging.get_logger(__name__) 


"""
In the following two methods ddim inversion and reconstruction is done. The formulas are the same mentioned in Eq.10 and Eq.12
of the paper. In forward_ddim(...) it is calling backward_ddim(...) because it is symmetric.

- alpha_t and alpha_tm1 are noise schedule at time t, to determine how much noise is present at step t
- eps_xt is the noise predicted at time t 
- x_t is the image at time t
"""
def backward_ddim(x_t, alpha_t: float, alpha_tm1: float, eps_xt):
    """ from noise to image"""
    return (
        alpha_tm1**0.5
        * (
            (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )

def forward_ddim(x_t, alpha_t: float, alpha_tp1: float, eps_xt):
    """ from image to noise, it's the same as backward_ddim"""
    return backward_ddim(x_t, alpha_t, alpha_tp1, eps_xt)


"""
They have redifined StableDiffusionPipeline to modify its behavior. Particularly to:
- add DDIM-based inversion/reconstruction (forward_diffusion, backward_diffusion)
- customize how prompts are embedded, latents are handled, and images are decoded
so any reference to StableDiffusionPipeline will refer to the modified version.
"""
class StableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]
        #safety_checker: StableDiffusionSafetyChecker = None
        #feature_extractor: CLIPFeatureExtractor = None
    ):
        super().__init__()

        self.register_modules( # registers the following components in the pipeline
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True) 
    
    """
    Transform the prompt into text embedding using CLIPTokenizer
    """
    @torch.inference_mode()
    def get_text_embedding(self, prompt):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        return text_embeddings
    

    """
    We obtain the latent vector of the input image of the VAE.
    First we retrieve the DiagonalGaussianDistribution, encoding_dist, that it uses it to sample from it or directly from the mean using .mode().
    Then the latents are scaled.
    """
    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist # as mentioned in the paper it's a zero-shot approach but we still need to know a VAE and UNet of a LDM
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    """
    Despite the name this function can do both backward and forward DDIM sampling. In our case it is used as inversion and reconstruction exploiting formulas defined above.
    """
    @torch.inference_mode()
    def backward_diffusion(
        self,
        use_old_emb_i=25, # set the inference step from which changes the text embedding to the modified one
        text_embeddings=None,
        old_text_embeddings=None, # not modified prompt
        new_text_embeddings=None, # modified prompt
        latents: Optional[torch.FloatTensor] = None, # initial latent input, x0 if reverse, x_T if reconstruction
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5, # strength of classifier-free guidance
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process: True = False,
        **kwargs,
    ):
        """ Generate image from text prompt and latents
        """
        do_classifier_free_guidance = guidance_scale > 1.0 # enables classifier-free guidance if guidance scale > 1.0
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma 
        # scales the input latents to the same noise level on which the model was trained on 
        # -> because VAE scales to a distribution, but might not be the same used in training, so we take it from the scheduler to be sure

        if old_text_embeddings is not None and new_text_embeddings is not None: # a bool to be sure to use the modified prompt from a certain inference step
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False


        for i, t in enumerate(self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            if prompt_to_prompt:
                if i < use_old_emb_i: # here it replaces the old one with the modified one
                    text_embeddings = old_text_embeddings
                else:
                    text_embeddings = new_text_embeddings

            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents # if we're using cfg, it duplicates the latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) 
            # So it scales the obtained latent based on the timestep. We must match the scale of the latents used during training, otherwise we would obtain garbage predictions
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings #predicting the noise guided by text embedding
            ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2) # it splits the noise prediction into two parts, one for uncod and the other cond
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            prev_timestep = ( # calculate the previous timestep index for DDIM update
                t
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps
            )

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
            
            # Here we compute alpha_t and alpha_t-1 (this latter one based on prev_timestep)
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            if reverse_process: # if we're doing the reverse process we will swap the directions, inverting variable values
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
            latents = backward_ddim( # perform reverse/reconstruction process
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )
        return latents

    
    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs) -> List[Image.Image]:
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image) -> List[Image.Image]: # this function was missing and giving error
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image


parser = argparse.ArgumentParser(description="Process images for DDIM.")
parser.add_argument("--target", type=str, help="Path to the folder containing images.")
parser.add_argument("--output", type=str, help="Path to the output folder for saving images.")
args = parser.parse_args()

output_folder = args.output
os.makedirs(output_folder, exist_ok=True)


pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
#pipe.scheduler = DDIMScheduler.from_config(path / "scheduler")
#pipe.scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler") 
#here I have modified the code inserting this, since the path is not working and they are already taking that stablediffusion pipeline, I took also its scheduler
pipe = pipe.to("cuda")

def numpy_to_pil(self, images):
    pil_images = []
    for image in images:
        image = (image * 255).round().astype("uint8")
        pil_images.append(Image.fromarray(image))
    return pil_images


def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")

    tform = transforms.Compose(
        [
            transforms.Resize((target_size,target_size)), #resizing to 512x512
            transforms.CenterCrop(target_size), #center cropping
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0

print(args)
images_path = Path(args.target)
#images_list = list(images_path.glob('*.[jp][pn]g'))
images_list = list(images_path.glob("*.jpg")) + \
              list(images_path.glob("*.jpeg")) + \
              list(images_path.glob("*.png"))
# images_list = list(images_path.glob('*'))
# images_list = [img for img in images_list if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]


from natsort import natsorted
images_list_sorted = natsorted(images_list)
images_list_str= [str(x) for x in images_list_sorted]

index = 0
nlp = spacy.load("en_core_web_sm")
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
image_size = 512
print("Loading BLIP model...")
blipmodel = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
print("BLIP model loaded.")
blipmodel.eval()
blipmodel = blipmodel.cuda()

print("Found", len(images_list_str), "images.")
print("images_list_str =", images_list_str)

for impath in images_list_str:
    print("Entering in for loop")
    img = load_img(impath).unsqueeze(0).to("cuda")
    print(index)
    
    prompt = blipmodel.generate(img, sample=True, num_beams=3, max_length=40, min_length=5)[0]
    
    prompt = replace_first_noun(prompt, 'big tree') # using big tree as replacement for every input is not optimal

    print(prompt)

    text_embeddings = pipe.get_text_embedding(prompt)

    rng_generator=torch.Generator(device=pipe.device).manual_seed(0)    

    image_latents = pipe.get_image_latents(img, rng_generator)

    reversed_latents = pipe.forward_diffusion(
        latents=image_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=999,
    )

    reconstructed_latents = pipe.backward_diffusion(
        latents=reversed_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=20,
    )

    # guidance_scale=1 so we follow the prompt only

    def latents_to_imgs(latents):
        x = pipe.decode_image(latents)
        x = pipe.torch_to_numpy(x)
        x = numpy_to_pil(x)
        return x

    image = latents_to_imgs(reconstructed_latents)[0]

    print(f"Saving image {index} to {os.path.join(output_folder, str(index) + '.png')}")
    image.save(os.path.join(output_folder, str(index) + '.png'), format="PNG")
    index += 1