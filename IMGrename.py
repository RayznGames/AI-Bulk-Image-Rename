#!/usr/bin/env python3
# Dependecies to install in your .venv
#    pip install torch pillow tqdm transformers

#Expensive load BLip2Model 
# (Replace [ "Load_Blip_Base(device) ] with this one)
"""
def load_blip2_model(device):
    #Loads the BLIP 2 image captioning model and its processor = “Salesforce/blip2-opt-6.7b-coco”.                
    model_name = "Salesforce/blip2-opt-6.7b-coco"      
    #Load processor and model
    logging.info("Loading BLIP-2 (image to text) pipeline…")
    processor = Blip2Processor.from_pretrained( model_name, CACHE_PATH, local_files_only= LOCAL_FILES )
    model = Blip2ForConditionalGeneration.from_pretrained( model_name, device = device,  cache_dir = CACHE_PATH, local_files_only= LOCAL_FILES).to(device)
    return processor, model
"""    

# Script -  
import argparse
import logging
import os
from pathlib import Path

import torch
from PIL import Image
import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration # - Smaller model
#from transformers import Blip2Processor, Blip2ForConditionalGeneration # - Bigger model

#Makes sure we download or use the local model files.
os.environ["TRANSFORMERS_OFFLINE"] = "0" #Change to 1 for offline mode
LOCAL_FILES:bool = False #False for downloading 

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".exr", ".tga"}
CACHE_PATH = "ModelFiles"
MAX_RENAME_ATTEMPTS = 1000
FILTER_START_OUTPUT = {"a ","an ","the " , "of a ", "image of a ", "illustration of a ", "black and white", "icon of a "}
PROMPT_TEXT = "This picture shows" 


#Sets up logging for the console.
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

def load_blip_base(device="cpu"):
    #Load the BLIP image captioning model 350m from Local drive- "Salesforce/blip-image-captioning-base"        
    model_name = "Salesforce/blip-image-captioning-base"    #The model name to load 
    #Prepare config
    config_cls = BlipForConditionalGeneration.config_class   #AutoConfig   
    cfg = config_cls.from_pretrained(
        model_name,
        cache_dir= CACHE_PATH,
        local_files_only=LOCAL_FILES, 
    )
    #Load processor and model
    logging.info("Loading BLIP (Lightweight) (image to text) pipeline…")
    processor = BlipProcessor.from_pretrained(model_name, CACHE_PATH, local_files_only= LOCAL_FILES)
    model = BlipForConditionalGeneration.from_pretrained(model_name, config= cfg, cache_dir= CACHE_PATH, local_files_only= LOCAL_FILES).to(device)
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
    # Replace whitespace with single underscore
    underscored = "_".join(text.split()) #we split by the spaces and join the splits with an underscore
    return underscored.strip("_")


def generate_caption(img_path, processor, model, device):
    """
    Returns the top‑1 caption for a given image.
    """
    try:
        image = Image.open(img_path).convert("RGB")       
    except Exception as e:
        logging.warning(f"Could not open {img_path}: {e}")
        return None    
       
    inputs = processor( images=image, text=PROMPT_TEXT, return_tensors="pt").to(device)
    with torch.no_grad():
        #generated_ids = model.generate(**inputs, max_length=50) 
        generated_ids = model.generate(**inputs, max_new_tokens=30,
                                        num_beams=30,         
                                        min_length=2,                      
                                        repetition_penalty=2.0,                                        
                                        no_repeat_ngram_size=20,
                                        diversity_penalty = 0.5,
                                        use_cache = False,    
                                        #Careful enabling theese (Improves quality)
                                        #custom_generate='transformers-community/group-beam-search',
                                        #num_beam_groups = 3,
                                        #num_return_sequences=3,
                                        #trust_remote_code=True, #Carefull for malixious code!!!
                                        early_stopping=True)
        
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)

    # Remove the prompt from output
    if caption.lower().startswith(PROMPT_TEXT.lower()):
        caption = caption[len(PROMPT_TEXT):].lstrip()    
    
    # Change or remove filter start output at your needs
    for item in FILTER_START_OUTPUT:
        if caption.lower().startswith(item.lower()):
            caption = caption[len(item):].lstrip()
    return caption.strip()


def safe_rename(src: Path, new_name_base: str):
    ext = src.suffix.lower()    
    f_candidate = src.parent / f"{new_name_base}{ext}"
    #Test final candidate existance
    if not f_candidate.exists():       
        os.rename(src,f_candidate)
        return f_candidate    
    #Duplicate name we need ad count num to rename!!
    for i in range(MAX_RENAME_ATTEMPTS):
        f_candidate = src.parent / f"{new_name_base}_{i}{ext}"
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

    #Processing tqdm progress bar 
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
