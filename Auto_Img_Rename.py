#!/usr/bin/env python3
"""
Image Renamer (BLIP version) - (BLIP 2 is hidden because is way more expensive and has more memory constraints) 

* Reads all image files in a directory.
* Generates a short caption for each image with the BLIP image to text pipeline.
* Sanitises that caption and uses it as the new file name avoiding duplicates.

"""
# Dependecies to install in your .venv
#    pip install torch pillow tqdm transformers

#Expensive load BLip2Model 
# (Replace [ "Load_Blip_Base(device) ] with this one)
"""
def load_blip2_model(device):
    #Loads the BLIP 2 image captioning model and its processor = “Salesforce/blip2-opt-6.7b-coco”.   
    cache_path = "ModelFiles"          
    model_name = "Salesforce/blip2-opt-6.7b-coco"      
    logging.info("Loading BLIP-2 (image to text) pipeline…")
    #Load processor and model
    processor = Blip2Processor.from_pretrained( model_name, cache_path, device = device,  )
    model = Blip2ForConditionalGeneration.from_pretrained( model_name, device = device,  cache_dir = cache_path).to(device)
    return processor, model
"""    

# Script -  
import argparse
import logging
import os
from pathlib import Path
#Makes sure we use only the contained AI model files within the local systm
os.environ["TRANSFORMERS_OFFLINE"] = "1"
#import re

import torch
from PIL import Image
import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration # - Smaller model
#from transformers import Blip2Processor, Blip2ForConditionalGeneration # - Bigger model

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".exr", ".tga"}
MAX_RENAME_ATTEMPTS = 1000
FILTER_START_OUTPUT = {"a ","an ","the " , "of a ", "image of a ", "illustration of a ", "black and white", "icon of a "}


#Sets up logging dor the console.
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

def load_blip_base(device="cpu"):
    #Load the BLIP image captioning model 350m from Local drive- "Salesforce/blip-image-captioning-base"        
    cache_path = "ModelFiles"   #Local AI model files path
    model_name = "Salesforce/blip-image-captioning-base"    #The model name to load 
    #Prepare config
    config_cls = BlipForConditionalGeneration.config_class   #AutoConfig
    cfg = config_cls.from_pretrained(
        model_name,
        cache_dir=cache_path,
        local_files_only=True,          # you already have the files locally
    )
    cfg.tie_word_embeddings = False
 
    logging.info("Loading BLIP (Lightweight) (image to text) pipeline…")
    #Load processor and model
    processor = BlipProcessor.from_pretrained(model_name, cache_path, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(model_name, config=cfg, cache_dir= cache_path, local_files_only=True).to(device)
    return processor, model

def get_image_paths(folder: Path):
    #if folder is not a directory raise exception! 
    if not folder.is_dir():
        logging.error(f"Provided path {folder} is not a directory.")
        raise SystemExit(1)

    #"Filtered list by extension of all image files in the provided path
    images = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(folder.rglob(f"*{ext}")) #Match by extension
    return sorted(images, key=lambda p: p.name.lower())

def sanitize_filename(text):
    # Replace whitespace (including multiple) with single underscore
    underscored = "_".join(text.split()) #we split by the spaces and join the splits with an underscore
    return underscored.strip("_") #or "untitled"


def generate_caption(img_path, processor, model, device):
    """
    Returns the top‑1 caption for a given image.
    """
    try:
        image = Image.open(img_path).convert("RGB")       
    except Exception as e:
        #logging.warning(f"Could not open {img_path}: {e}")
        return None
    
    prompt_text = "This picture shows"    
    inputs = processor( images=image, text=prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        #generated_ids = model.generate(**inputs, max_length=50) 
        generated_ids = model.generate(**inputs, max_new_tokens=20,
                                        num_beams=25,         
                                        min_length=2,                      
                                        repetition_penalty=1.75,                                        
                                        no_repeat_ngram_size=20,                                        
                                        early_stopping=True)
        
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)

    # Remove the prompt words (case‑insensitive)
    if caption.lower().startswith(prompt_text.lower()):
        caption = caption[len(prompt_text):].lstrip()    
    
    # Change or remove filter start output at your needs
    for item in FILTER_START_OUTPUT:
        if caption.lower().startswith(item.lower()):
            caption = caption[len(item):].lstrip()
    return caption.strip()


def safe_rename(src: Path, new_name_base: str):
    ext = src.suffix.lower()
    new_name = f"{new_name_base}"   
    f_candidate = src.parent / f"{new_name}{ext}"
    #Test final candidate existance
    if not f_candidate.exists():       
        os.rename(src,f_candidate)
        return f_candidate    
    #Duplicate name we need ad count num to rename!!
    for i in range(MAX_RENAME_ATTEMPTS):
        f_candidate = src.parent / f"{new_name}_{i}{ext}"
        if not f_candidate.exists():
            try:
                os.rename(src, f_candidate)
                return f_candidate
            except Exception as e:
                logging.warning(f"Rename failed: {src} -> {f_candidate}: {e}")
                break               
  
    # Final fallback – keep original name
    logging.info(f"Keeping original name for {src.name}")
    return src


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")       
    processor, model = load_blip_base(device) #Change this to - load_blip2_model(device) for the big model

    folder = Path(args.folder).expanduser().resolve()
    image_paths = get_image_paths(folder)
    if not image_paths:
        logging.warning("No supported images found.")
        return

    rename_log = []

    #Processing tqdm bar 
    for img_path in tqdm.tqdm(image_paths, desc="Inferencing images"):
        caption = generate_caption(img_path, processor, model, device)

        if caption is None or caption == "":
            sanitized = img_path.stem  # keep original name
        else:
            sanitized = sanitize_filename(caption)
            if sanitized == "":
                sanitized = img_path.stem #fallback

        #For rename    
        new_path = safe_rename(img_path, sanitized)
        rename_log.append((img_path.name, new_path.name))

        # For just logging
        #rename_log.append((img_path.name, sanitized))       

    # Final summary
    print("\nAI Rename summary:")
    for old, new in rename_log:
        print(f"  {old} -> {new}")

# Modify your inputs if you want extra control on command over the generation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename images based on BLIP generated captions."
    )
    parser.add_argument("folder", help="Path to the folder containing images.")
    args = parser.parse_args()
    setup_logging()
    main(args)
