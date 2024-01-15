#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Yuan Gao
Created Time:
Note: Part of codes are inspired by paper "Any-to-Any Generation via Composable Diffusion"
'''

import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
from core.models.model_module_infer import model_module
from PIL import Image
from IPython.display import Audio
# from IPython.display import Image
from IPython.display import Video
import torchaudio
import torch


proxies = {
    'http': '127.0.0.1:7890',
    'https': '127.0.0.1:7890',
}


model_load_paths = ['CoDi_encoders.pth', 'CoDi_text_diffuser.pth', 'CoDi_audio_diffuser_m.pth', 'CoDi_video_diffuser_8frames.pth']
inference_tester = model_module(data_dir='checkpoints/', pth=model_load_paths)
inference_tester = inference_tester.cuda()
inference_tester = inference_tester.eval()


# # ---------text 2 image----------------
# prompt = "A beautiful oil painting of a birch tree standing in a spring meadow with pink flowers, a distant mountain towers over the field in the distance. Artwork by Alena Aenami"
# prompt = 'a dark-skinned boy wearing a knitted woolly hat and a light and dark grey striped jumper with a grey zip, leaning on a grey wall.'
prompt = "Bionic man dreams of electronic sheep."

#
# Generate image
image = inference_tester.inference(
                xtype = ['image'],              # the type of data you desire
                condition = [prompt],
                condition_types = ['text'],     # the type of input data
                n_samples = 5,                  # the number of images you want to generate
                image_size = 256,               # the size of the generated image
                ddim_steps = 50)                # steps, related to time cost


# Load an image input---------image 2 text----------
im = Image.open('./data/iaprtc12/images/03/3111.jpg').resize((224, 224))
# # im.show()

prompt = 'a dark-skinned boy wearing a knitted woolly hat and a light and dark grey striped jumper with a grey zip, leaning on a grey wall.'

# Generate image
text = inference_tester.inference(
                xtype = ['text'],
                condition = [prompt],
                condition_types = ['text'],
                n_samples = 1,
                image_size = 256,
                ddim_steps = 50)



# -----------audio 2  text----------------
path = './assets/demo_files/train_sound.flac'

audio_wavs, sr = torchaudio.load(path)
audio_wavs = torchaudio.functional.resample(waveform=audio_wavs, orig_freq=sr, new_freq=16000).mean(0)[:int(16000 * 10.23)]
Audio(audio_wavs.squeeze(), rate=16000)
n_samples = 4
text = inference_tester.inference(
                xtype = ['text'],
                condition = [audio_wavs],
                condition_types = ['audio'],
                n_samples = n_samples,
                ddim_steps = 100,
                scale = 7.5)
# print(text)


# ---------------text 2 video---------------------
# Give A Prompt
prompt = 'Fireworks bursting in mid-air.'
# prompt = 'A small plant emerges from the soil and the plant grows and eventually becomes rose.' Little boy picks up rose from table, cartoon style.

n_samples = 1
outputs = inference_tester.inference(
                    ['video'],
                    condition = [prompt],
                    condition_types = ['text'],
                    n_samples = 1,
                    image_size = 256,
                    ddim_steps = 50,
                    num_frames = 8,
                    scale = 7.5)

video = outputs[0][0]

# ---------------video 2 text---------------------
# Load a video, note that it is best to input video below 10s
from core.common.utils import load_video

video_path = './assets/demo_files/cloud.mp4'
video = load_video(video_path, sample_duration=10.0, num_frames=8)

text = inference_tester.inference(
                xtype = ['text'],
                condition = [video],
                condition_types = ['video'],
                n_samples = 4,
                scale = 7.5,)
print(text[0])


# ---------------image 2 audio------------------
# Load an image
from PIL import Image
im = Image.open('./assets/demo_files/rain_on_tree.jpg')
# im.show()
# Generate audio
audio_wave = inference_tester.inference(
                xtype = ['audio'],
                condition = [im],
                condition_types = ['image'],
                scale = 7.5,
                n_samples = 1,
                ddim_steps = 50)[0]

# Play audio
from IPython.display import Audio
Audio(audio_wave.squeeze(), rate=16000)


# ---------------audio 2 image----------------------
pad_time = 10.23

path = './assets/demo_files/wind_chimes.wav'

audio_wavs, sr = torchaudio.load(path)
audio_wavs = torchaudio.functional.resample(waveform=audio_wavs, orig_freq=sr, new_freq=16000).mean(0)[:int(16000 * pad_time)]
padding = torch.zeros([int(16000 * pad_time) - audio_wavs.size(0)])
audio_wavs = torch.cat([audio_wavs, padding], 0)

# Audio(path, rate=16000)

# Generate image
images = inference_tester.inference(
                xtype = ['image'],
                condition = [audio_wavs],
                condition_types = ['audio'],
                scale = 7.5,
                image_size = 256,
                ddim_steps = 50)
images[0][0].show()



