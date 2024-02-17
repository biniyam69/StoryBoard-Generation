import torch
from PIL import Image
from io import BytesIO
import os
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionImg2ImgPipeline
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import replicate
from typing import Literal, Optional, Tuple


model_id = "stabilityai/stable-diffusion-2"
device = 'cuda:0'
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder='scheduler')
text2img_pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16).to(device)
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)



def gen_img2img(prompt: str, performance_selection: Literal['Speed', 'Quality', 'Extreme Speed'] = "Extreme Speed", 
                            aspect_ratios_selection: str = "1024,1024",image_seed: int = 1234, sharpness: int = 2, cn_img1: str =  "Img", cn_type1= "ImagePrompt", 
                            ):
    pass

img2img_replicate = gen_img2img()
    

# def generate_images(df, out_dir):
#     for index, row in df.iterrows():
#         pass

# prefix = "Generate an artistic interpretation of the text "
# prefix = ""

def generate_images(df, out_dir):
    for index, row in df.iterrows():
        prompts = row["summarized_story"]
        story_type = row["story_type"]
        if len(prompts)>0:
            init_prompt = prompts[0]
            init_image = text2img_pipe(init_prompt).images[0]
            final_index = story_type+"/"+row["story_name"]+"_segment_0"
            save_path = os.path.join(out_dir,str(final_index) +'.jpg')
            init_image.save(save_path)
            
            for idx, prompt in enumerate(prompts[1:]):    
                images = img2img_pipe(prompt=prompt, image=init_image, strength=0.90, guidance_scale=7.5).images
                final_index = story_type+"/"+row["story_name"]+"_segment_"+ str(idx+1)
                save_path = os.path.join(out_dir,str(final_index) +'.jpg')
                images[0].save(save_path)


def generate_images_baseline0(prompts, prefix=""):
    generated_images = []
    if len(prompts)>0:
        for idx, prompt in enumerate(prompts):    
            image = text2img_pipe(prefix+prompt).images[0]            
            generated_images.append(image)
    return generated_images


def generate_images_baseline(prompts, strength, guidance_scale, prefix=""):
    if len(prompts)>0:
        init_prompt = prompts[0]
        init_image = text2img_pipe(prefix+init_prompt).images[0]
        generated_images = [init_image]
        for idx, prompt in enumerate(prompts[1:]):    
            images = img2img_pipe(prompt=prefix+prompt, image=init_image, strength=strength, guidance_scale=guidance_scale).images
            generated_images.append(images[0])
    return generated_images

def bert_sentence_embedding(sentence):
    tokens = bert_tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**tokens.to(device))
    # Extract the embeddings for the [CLS] token
    cls_embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embedding

def findSimilarPromptBert(query, string_list):
    if not string_list:
        return None
    query_embedding = bert_sentence_embedding(query)
    string_embeddings = [bert_sentence_embedding(s) for s in string_list]

    similarities = [cosine_similarity(query_embedding, s.reshape(1, -1))[0, 0] for s in string_embeddings]
    most_similar_index = similarities.index(max(similarities))
    return most_similar_index


def generate_images_proposed(story, strength, guidance_scale, prefix=""):
    init_prompt = story[0]
    # print(init_prompt)
    init_image = text2img_pipe(prefix+init_prompt).images[0]
    generated_images = [init_image]
    for idx, prompt in enumerate(story[1:]):
        most_similar_index = findSimilarPromptBert(prompt, story[:idx+1])
        print("CURRENT PROMPT:",prompt)
        print("SIMILAR PROMPT", story[most_similar_index])
        images = img2img_pipe(prompt=prefix+prompt, image=generated_images[most_similar_index], strength=strength, guidance_scale=guidance_scale).images
        generated_images.append(images[0])
    return generated_images



