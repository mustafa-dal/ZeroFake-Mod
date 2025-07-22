print("Importing required libraries...\n")

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torch.nn.functional as nnf

from diffusers import DDIMScheduler, StableDiffusionPipeline
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
from diffusers.utils import deprecate, logging, numpy_to_pil
from natsort import natsorted

from PIL import Image
import requests
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from blipmodels import blip_decoder
import spacy

import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE

import os
import sys
import argparse

import nltk
from nltk.corpus import brown
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk import download

import zipfile
import random
import gdown

import pandas as pd

import nltk
nltk.download('punkt_tab')
from nltk.corpus import wordnet as wn
download('wordnet')
download('omw-1.4')

from sentence_transformers import SentenceTransformer, util

from torchvision.models.inception import inception_v3
from scipy.linalg import sqrtm
import torch.nn as nn
from torcheval.metrics import FrechetInceptionDistance

print("Now loading helper functions and models...\n")

### HELPER FUNCTIONS

def replace_first_noun(sentence, replacement):
    doc = nlp(sentence)

    first_chunk = None
    for chunk in doc.noun_chunks:
        if first_chunk is None or chunk.start_char < first_chunk.start_char:
            first_chunk = chunk
    # This preliminary part of the function is essential because noun chunks are not sorted by textual order, so it may happen that a noun that is at the end
    # of the phrase is taken

    if first_chunk:
        if first_chunk[0].pos_ == "DET": # If the first token is a determiner (like "the", "a", "an"), preserve it
            noun_start = first_chunk[1].idx # the start is shifted
            noun_end = first_chunk.end_char
            return sentence[:noun_start] + replacement + sentence[noun_end:]
        else:
            return sentence[:first_chunk.start_char] + replacement + sentence[first_chunk.end_char:]

    return sentence
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
    return 2.0 * image - 1.0 # converting the image into [-1,1] range

### REDEFINITION OF STABLEDIFFUSIONPIPELINE

class StableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        #safety_checker: StableDiffusionSafetyChecker = None,
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
    First we retrieve the DiagonalGaussianDistribution (encoding_dist) that it uses it to sample from it or directly from the mean using .mode().
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
        """
        Generate image from text prompt and latents
        """
        do_classifier_free_guidance = guidance_scale > 1.0 # enables classifier-free guidance if guidance scale > 1.0
        self.scheduler.set_timesteps(num_inference_steps)
        # initialize values of alphas, based on the number of inference steps, so we separate 1k training steps based on num_inference_steps
        # 1k, if we have 50 as inference steps, we will have [980, 960, ..., 0] for example
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma
        # scales the input latents to the same noise level on which the model was trained on
        # -> because VAE scales to a distribution, but might not be the same used in training, so we take it from the scheduler to be sure

        if old_text_embeddings is not None and new_text_embeddings is not None: # a bool to be sure to use the modified prompt from a certain inference step
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False


        for i, t in enumerate(self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
        # this is to show a progress bar when we do inversion process
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
            # the ratio gives the interval between steps in order to retrieve the previous timestep as a subtraction with the current t.
            # num_train_timeteps is the number of diffusion steps used during training of the diffusion model (typically 1k), that defines how gradually noise is added in forward process.
            # During inference you try to do 1k in 50 inference_steps instead. So those 50 steps are spaced within the 1k-step training schedule

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

            # Here we compute alpha_t and alpha_t-1 (this latter one based on prev_timestep)
            alpha_prod_t = self.scheduler.alphas_cumprod[t] # this is the overline_alpha_t, the cumulative product of alphas up to time t
            alpha_prod_t_prev = ( # we calculate overline_alpha_t-1, setting it to the previous timestep (an approximation of what it was)
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod # or directly to the last one
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
    
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("cuda")

print("Dataset download started...\n")
    
### IMAGE FOLDER CREATION AND DATASET DOWNLOAD

parser = argparse.ArgumentParser(description="Extract n images from the dataset starting from a given index.")
parser.add_argument("--start", type=int, default=0, help="Start index for image selection.")
parser.add_argument("--n", type=int, default=20, help="Number of images to extract.")
parser.add_argument("--mode",type=str,default='r',help="Select fake (f) or real (r) images")
args = parser.parse_args()

output_folder = Path("/home/mbrigo/ZeroFake-Mod/reconstructed")
images_path = Path("/home/mbrigo/ZeroFake-Mod/imgs")

file_id = "1QLYJMhy0CbBVT01BLkkw7KPPL5BpmxnH"
url = f"https://drive.google.com/uc?id={file_id}"
output = "Chameleon.zip"

gdown.download(url, output, quiet=False)

zip_path = "Chameleon.zip"
extract_to = images_path
final_part_path_images = "Chameleon/test/0_real/"
n = args.n
start = args.start
mode = args.mode

if(mode == 'f'): final_part_path_images = "Chameleon/test/1_fake/"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Filter only files in 1_fake or 0_real
    real_files = [f for f in zip_ref.namelist() if final_part_path_images in f and not f.endswith("/")]

    #selected_files = random.sample(real_files, n)
    selected_files = real_files[start:start+n]

    for file in selected_files:
        zip_ref.extract(file, extract_to)
        
#dataset_images_path = images_path / "Chameleon" / "test" / "0_real"
dataset_images_path = images_path / Path(final_part_path_images)
#real_images = natsorted(dataset_images_path.glob("*.*"))[:20] # to try putting selected_files here instead of [:20] here
real_images = natsorted([images_path / Path(f) for f in selected_files])
images_list_str= [str(x) for x in real_images]

print("Dataset downloaded successfully!\n")


### LOADING BLIP MODEL
index = 0
nlp = spacy.load("en_core_web_sm")
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
image_size = 512
print("Loading BLIP model...\n")
blipmodel = blip_decoder(pretrained=model_url, image_size=image_size, vit='base') # Error solved when using generate with this, that can be found in blip.py and med.py
print("BLIP model loaded.\n")
blipmodel.eval()
blipmodel = blipmodel.cuda()

print("Found", len(images_list_str), "images.")


### MOST FREQUENT WORDS IN ENGLISH

print("Importing most common words for adversarial prompts...\n")
word_path = Path("/home/mbrigo/ZeroFake-Mod/words/most-common-nouns-english.csv")
df = pd.read_csv(word_path)
df.head()

df = df['Word'].str.lower()
df = df[df.str.len() > 2] #filter out elements that have len less than 2, because it was also containing articles
df = df[~df.str.contains(r'[^\w\s]', regex=True)] #filter out elements that contain punctuation

def is_concrete_noun(word):
    synsets = wn.synsets(word, pos=wn.NOUN)
    for syn in synsets:
        for path in syn.hypernym_paths():
            if any(h.name().startswith("physical_entity") for h in path):
                return True
    return False
df = df[df.apply(is_concrete_noun)]

noun_list = [noun for noun in df]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_perturbed_prompts(sentence): return [replace_first_noun(sentence,noun) for noun in noun_list]

### ALL-MINILM 

print("Installing context-aware embedding model...\n")
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

### DDIM INVERSION AND RECONSTRUCTION

print("Starting DDIM inversion + reconstruction processes...\n")
for impath in images_list_str:

    img = load_img(impath).unsqueeze(0).to("cuda")
    print(index)

    prompt = blipmodel.generate(img, sample=True, num_beams=3, max_length=40, min_length=5)[0]

    print(prompt)

    candidate_prompts = get_perturbed_prompts(prompt)

    #original_prompt = sentence_embedding(prompt) # to be changed with sentence_transformer.encode(prompt)
    original_prompt = sentence_transformer.encode(prompt)
    min_sim = 1.0
    selected = None
    for adv in candidate_prompts:
        #adv_prompt_word_vector = sentence_embedding(adv) # to be changed with sentence_transformer.encode(adv)
        adv_prompt_word_vector = sentence_transformer.encode(adv)
        #res = cosine_similarity(adv_prompt_word_vector,original_prompt) # to be changed with sentence_transformer.similarities(adv_prompt_word_vector,original_prompt)
        res = util.cos_sim(adv_prompt_word_vector,original_prompt)
        if res < min_sim:
            min_sim = res
            selected = adv
        #print(f"Candidate prompt: {adv} with cosine similarity score {res}")
    
    if selected is None: selected = 'An image'

    prompt = selected

    #prompt = replace_first_noun(prompt, target_word)

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
    
### FRECHET INCEPTION DISTANCE

print("Loading FID distance...\n")
inception_model = inception_v3(pretrained=True, transform_input=False).eval()
inception_model.fc = nn.Identity()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception_model.to(device)

def FID_preprocess(img):
    img = cv2.resize(img, (299, 299))
    img_tensor = torch.from_numpy(img).float()
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    return img_tensor

def extract_features(img):
    img_tensor = FID_preprocess(img)

    # FID implemenetation of torcheval already normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    img_tensor = (img_tensor - mean) / std

    with torch.no_grad():
        features = inception_model(img_tensor)

    return features.squeeze().cpu().numpy()

def feature_distance(feat1, feat2):
    return np.sum((feat1 - feat2)**2)

fid = FrechetInceptionDistance().to(device)

### COMPARISON CYCLE

print("Trying to initiate comparison cycle...\n")

folder1_path = dataset_images_path

folder2_path = output_folder

output_file = "testfake.txt"

mean_ssim_score = []
mean_feature_distance_score = []

with open(output_file, "w") as f:
    f.write("Image Filename\tPixel Similarity\n")
    index = 0
    for image_path1, image_path2 in zip(natsorted(Path(folder1_path).glob("*?.*")), natsorted(Path(folder2_path).glob("*?.*"))):

        print(image_path1)
        print(image_path2)
        image1_orig = cv2.imread(str(image_path1))
        image2_orig = cv2.imread(str(image_path2))

        image1 = cv2.resize(image1_orig, (512, 512))
        image2 = cv2.resize(image2_orig, (512, 512))

        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # frechet_score = frechet_distance_score(gray1,gray2)
        # print(f"Fréchet score: {frechet_score}")

        features1 = extract_features(image1_orig)
        features2 = extract_features(image2_orig)
        feature_dist = feature_distance(features1,features2)
        mean_feature_distance_score.append(feature_dist)
        print("Feature distance score:", feature_dist,"\n")
        f.write(f"Feature distance score of {index}: \t{feature_dist}\n")

        image1_tensor = FID_preprocess(image1_orig)
        image2_tensor = FID_preprocess(image2_orig)
        fid.update(image1_tensor, is_real=True)
        fid.update(image2_tensor, is_real=False)

        ssim_score = ssim(gray1, gray2)

        f.write(f"SSIM score of {index}: \t{ssim_score}\n")
        print("SSIM score:", ssim_score,"\n")
        index+=1
        mean_ssim_score.append(ssim_score)

    fid_score = fid.compute()
    f.write(f"FID score \t{fid_score.item()}\n")
    print("FID score \t", fid_score.item(),"\n")
    f.write(f"Mean feature distance \t{np.mean(mean_feature_distance_score)}\n")
    print(f"Mean feature distance \t{np.mean(mean_feature_distance_score)}\n")
    f.write(f"Mean SSIM score \t{np.mean(mean_ssim_score)}\n")
    print(f"Mean SSIM score \t{np.mean(mean_ssim_score)}\n")

print("Results saved to", output_file)