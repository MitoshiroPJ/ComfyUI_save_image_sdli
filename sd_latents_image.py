from __future__ import annotations

import io
import os
import json
import random
import re
import time

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args

import folder_paths
import numpy as np
import json

import torch
import safetensors.torch

from nodes import VAELoader
import comfy.sd

REDUCTION_RATIO = ["1/1", "1/2", "1/4", "1/8"]
LATENT_TYPE = ["SD1", "SDXL", "SD3", "FLUX.1"]


def get_save_image_path(filename_prefix: str, output_dir: str, image_width=0, image_height=0) -> tuple[str, str, int, str, str]:

    def compute_vars(input: str, image_width: int, image_height: int) -> str:
        input = input.replace("%width%", str(image_width))
        input = input.replace("%height%", str(image_height))
        now = time.localtime()
        input = input.replace("%year%", str(now.tm_year))
        input = input.replace("%month%", str(now.tm_mon).zfill(2))
        input = input.replace("%day%", str(now.tm_mday).zfill(2))
        input = input.replace("%hour%", str(now.tm_hour).zfill(2))
        input = input.replace("%minute%", str(now.tm_min).zfill(2))
        input = input.replace("%second%", str(now.tm_sec).zfill(2))
        return input

    def get_counter(filename: str) -> int:
      try:
        match = re.match(rf'{filename_prefix}_(\d+)', filename)
        if match:
          return int(match.group(1))
        return 0
      except:
        return 0

    if "%" in filename_prefix:
        filename_prefix = compute_vars(filename_prefix, image_width, image_height)

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))

    full_output_folder = os.path.join(output_dir, subfolder)

    if os.path.commonpath((output_dir, os.path.abspath(full_output_folder))) != output_dir:
        err = "**** ERROR: Saving image outside the output folder is not allowed." + \
              "\n full_output_folder: " + os.path.abspath(full_output_folder) + \
              "\n         output_dir: " + output_dir + \
              "\n         commonpath: " + os.path.commonpath((output_dir, os.path.abspath(full_output_folder)))
        raise Exception(err)

    try:
        counter = max([get_counter(f) for f in os.listdir(full_output_folder) if f.startswith(filename_prefix)]) + 1
        #counter = max(filter(lambda a: os.path.normcase(a[1][:-1]) == os.path.normcase(filename) and a[1][-1] == "_", map(map_filename, os.listdir(full_output_folder))))[0] + 1
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return full_output_folder, filename, counter, subfolder, filename_prefix


def read_riff_chunk(file):
  chunk_id = file.read(4).decode('ascii')
  chunk_size = int.from_bytes(file.read(4), 'little')

  chunk_data = file.read(chunk_size)
  return (chunk_id, chunk_size, chunk_data)

def write_riff_chunk(file, chunk_id, chunk_data):
  file.write(chunk_id.encode('ascii'))
  file.write(len(chunk_data).to_bytes(4, 'little'))
  file.write(chunk_data)

def load_riff_file(file, size):
  _riff_header = file.read(4)
  _riff_size = int.from_bytes(file.read(4), 'little')
  riff_type = file.read(4)
  chunks = []
  while file.tell() < size:
    chunks.append(read_riff_chunk(file))
  return (riff_type, chunks)
  
def save_riff_file(file, riff_type, chunks):
  size = 4
  for chunk_id, chunk_size, chunk_data in chunks:
    size += 8 + chunk_size
  file.write(b'RIFF')
  file.write(size.to_bytes(4, 'little'))
  file.write(riff_type)
  for chunk_id, chunk_size, chunk_data in chunks:
    write_riff_chunk(file, chunk_id, chunk_data)

def get_prompt_text(prompt):
  for v in prompt.values():
    if v['class_type'].startswith('CLIPTextEncode'):
      return v['inputs']['text']
  return ''

def save_image(path, img, latents, safetensors_metadata, metadata, quality):
  # build safetensors data
  safetensors_dict = {
    'latents': latents.to(torch.float16)
  }

  safetensors_data = safetensors.torch.save(safetensors_dict, safetensors_metadata)

  # save the image as WEBP
  with io.BytesIO() as buffer:
    img.save(buffer, format="webp", quality=quality, exif=metadata)
    size = buffer.tell()
    buffer.seek(0)
    riff_type, chunks = load_riff_file(buffer, size)

  chunks.append(('SDLI', len(safetensors_data), safetensors_data))
  
  with open(path, 'wb') as f:
    save_riff_file(f, riff_type, chunks)

class SaveSdlImage:
  def __init__(self):
    self.output_dir = folder_paths.get_output_directory()
    self.type = "output"
    self.prefix_append = ""

  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "samples": ("LATENT", {"tooltip": "The latent to be decoded."}),
        "latent_type": (LATENT_TYPE, {"default": "SDXL"}),
        "reduction_ratio": (REDUCTION_RATIO, {"default": "1/1"}),
        "quality": ("INT", {"default": 80}),
        "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
      },
      "optional": {
        "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."}),
        "positive_prompt": ("STRING", {"default": "" }),
        "negative_prompt": ("STRING", {"default": "" })
      },
      "hidden": {
        "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
      },
    }

  RETURN_TYPES = ()
  FUNCTION = "save_sdlimages"

  OUTPUT_NODE = True

  CATEGORY = "image"
  DESCRIPTION = "Saves the input latents as SDLI format"

  def decode_latent(self, latent, vae, reduction_ratio):
    image = vae.decode(latent)[0]
    i = 255. * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    width = img.width
    height = img.height

    if reduction_ratio == '1/2':
      img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    elif reduction_ratio == '1/4':
      img = img.resize((img.width // 4, img.height // 4), Image.LANCZOS)
    elif reduction_ratio == '1/8':
      img = img.resize((img.width // 8, img.height // 8), Image.LANCZOS)  
    return (img, width, height)
  

  def save_sdlimages(self, samples, latent_type, reduction_ratio, quality, filename_prefix="ComfyUI",
                     vae=None, positive_prompt=None, negative_prompt=None, prompt=None, extra_pnginfo=None):
    filename_prefix += self.prefix_append
    full_output_folder, filename, counter, subfolder, filename_prefix = get_save_image_path(filename_prefix, self.output_dir)
    results = list()

    if positive_prompt is None or len(positive_prompt) == 0:
      if prompt is not None:
        positive_prompt = get_prompt_text(prompt)

    if vae is None:
      if latent_type == 'SD1':
        sd = VAELoader.load_taesd('taesd')
      elif latent_type == 'SDXL':
        sd = VAELoader.load_taesd('taesdxl')
      elif latent_type == 'SD3':
        sd = VAELoader.load_taesd('taesd3')
      elif latent_type == 'FLUX.1':
        sd = VAELoader.load_taesd('taef1')
      vae = comfy.sd.VAE(sd=sd)

    count = samples["samples"].shape[0]

    for idx in range(count):
      latents = samples["samples"][idx:idx+1]
      (img, width, height) = self.decode_latent(latents, vae, reduction_ratio)
      metadata = None

      if not args.disable_metadata:
        img_exif = img.getexif()
        if prompt is not None:
          img_exif[0x010f] = "Prompt:" + json.dumps(prompt)
        if extra_pnginfo is not None:
          workflow_metadata = ''
          for x in extra_pnginfo:
            workflow_metadata += json.dumps(extra_pnginfo[x])
          img_exif[0x010e] = "Workflow:" + workflow_metadata
        metadata = img_exif.tobytes()

      filename_with_batch_num = filename.replace("%batch_num%", str(idx))
      file = f"{filename_with_batch_num}_{counter:05}.webp"
      path = os.path.join(full_output_folder, file)

      safetensors_metadata = {
        'latent_type': latent_type,
        'width': str(width),
        'height': str(height)
      }
      if positive_prompt is not None:
        safetensors_metadata['positive_prompt'] = positive_prompt
      if negative_prompt is not None:
        safetensors_metadata['negative_prompt'] = negative_prompt

      save_image(path, img, latents.cpu(), safetensors_metadata, metadata, quality)
      results.append({
        "filename": file,
        "subfolder": subfolder,
        "type": self.type
      })
      counter += 1

    return { "ui": { "images": results } }

class PreviewSdlImage(SaveSdlImage):
  def __init__(self):
    self.output_dir = folder_paths.get_temp_directory()
    self.type = "temp"
    self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "samples": ("LATENT", {"tooltip": "The latent to be decoded."}),
        "latent_type": (LATENT_TYPE, {"default": "SDXL"}),
        "reduction_ratio": (REDUCTION_RATIO, {"default": "1/1"}),
        "quality": ("INT", {"default": 80})
      },
      "optional": {
        "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."}),
        "positive_prompt": ("STRING", {"default": "" }),
        "negative_prompt": ("STRING", {"default": "" })
      },
      "hidden": {
        "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
      },
    }

NODE_CLASS_MAPPINGS = {
    "SaveSdlImage": SaveSdlImage,
    "PreviewSdlImage": PreviewSdlImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveSdlImage": "Save SDLI Image",
    "PreviewSdlImage": "Preview SDLI Image",
}