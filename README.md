## Why forking the repo?
This forked repo contains:
- a python file merging the originals `uni-ddim-inversion.py` and `sim.py`
- a notebook called `ddim_inversion.ipynb` that includes `uni-ddim-inversion.py` and `sim.py`
Both files apply the approach specified in the paper, however the first one is the latest version with the latest modifications. The notebook was initially built to address a resource constraints problem, since, as the author mentions in its paper, this zero-shot method is computationally expensive. Having a notebook allowed me to run this into Google Colab exploiting the limited usage of T4 GPU.
In order to make everything work it is important to change paths to `/blipmodels/` directory and also in the python/notebook file to match your directories.

## How to use it?
Load your images (or use the imported dataset) and change the variables `output_folder` and `images_path` accordingly (as long as other related paths) and run the python file/notebook, then your images will be reconstructed. 

So when executing the following python command line the test will be performed on that dataset. In order to change the dataset you have to modify the part of the code regarding the dataset in the file `inversion_sim.py`. It will work for the images has "decoded" in their names for the mode 'c'.

Run the following command to execute the python file:
```
# Example: Start from image 10, process 5 images, using real images 
python ZeroFake-Mod/inversion_sim.py --start 10 --n 5 --mode r

# Example: Start from the first image, process 20 compressed images
python ZeroFake-Mod/inversion_sim.py --start 0 --n 20 --mode c

--start  INT   Index of the first image to process (0-based).
--n      INT   Number of images to process starting from --start.
--mode   STR   'r' for real images, 'c' for compressed images.
```

## Improvements
- The function `replace_first_noun(…)` was sometimes missing the first noun to take resulting into a poor adversarial prompt. For example I noticed that it was considering *“pope”* as proper noun, so I decided to insert the detection also of proper noun(s) and plural nouns. A successive problem was the detection of compound nouns, so reading the documentation of spaCy I find out the `noun_chunks`, that divide phrases into noun chunks (not in textual order).
- In the paper they cite that they have used a hand-crafted list of nouns that ensure that the adversial prompt is helping deceiving the reconstruction and inversion. Nevertheless this list is missing and was only present a single word *“the big tree”* in the code given. For this reason I curated a list of most common english nouns filtering out articles, verbs, abstract nouns, words with punctuation and words with less than 3 characters.
- In the original code a way to take an adversarial prompt that was minimizing the cosine similarity between the original prompt and the adversarial one was not present. For this reason I imported a small `sentence-transformer` model to ensure a context-aware replacement. Then len(noun_list) prompts are generated replacing, as mentioned in the paper, the first noun that appears with each of the most common nouns. Cosine similarity of the original prompt with respect to the perturbed one is calculated, taking the one with the minimum value (most divergent).
- Testing the approach using the SSIM measure with other datasets (since the datasets used in the paper are not disclosed) showed that SSIM was giving false results about the reconstruction's quality. For this reason an additional FID measurement (to confirm the fact that real images are reconstructed worse than fake images) and a simple feature distance measure was applied (captured with Inception V3)
