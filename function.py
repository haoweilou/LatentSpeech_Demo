import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from pypinyin import pinyin, Style

def saveLog(log_record:dict,name="log",root="./save/log/"):
    log = pd.DataFrame(log_record)
    log.to_csv(f"{root}{name}")

def load_audio(file_path):
    # Load audio file
    waveform, sample_rate = torchaudio.load(file_path)
    
    return waveform, sample_rate

def melspectrogram(waveform, sample_rate, n_mels=80, n_fft=1024):
    # Create a MelSpectrogram transform
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft
    )
    
    # Apply the transform to the waveform
    mel_spectrogram = mel_spectrogram_transform(waveform)
    mel_spectrogram = torch.transpose(mel_spectrogram,1,2)
    return mel_spectrogram

def resample(waveform,  target_rate,start_rate = 16*1000):
    # Resample the waveform to the target sample rate
    waveform_resampled = torchaudio.transforms.Resample(
        orig_freq=start_rate,
        new_freq=target_rate
    )(waveform)
    return waveform_resampled

def slide_window(sequence,window_size=100,step=100):
    if len(sequence.shape) > 1: sequence = sequence[0]
    output = []
    for start in range(0, len(sequence) - window_size + 1, step):
        segment = sequence[start:start+step]
        output.append(segment)
    
    return np.array([output])

def saveModel(model,name,root="./save/model"):
    path = f"{root}/{name}.pth" 
    torch.save(model.state_dict(), path) 

def learning_rate(d_model=256,step=1,warmup_steps=400):
    return (1/math.sqrt(d_model)) * min(1/math.sqrt(step),step*warmup_steps**-1.5)

def loadModel(model,name,root="/home/haoweilou/scratch/model/denoise"):
    path = f"{root}/{name}.pth" 
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def save_audio(audio,sample_rate,name="",root="./sample/"):
    torchaudio.save(f'{root}/{name}.wav', audio, sample_rate)

def draw_wave(audio,name="Test",root="./fig/"):
    #audio: t
    t = torch.linspace(0, audio.shape[-1], audio.shape[-1])
    plt.figure(figsize=(10, 8))
    # # Convert the tensors to numpy arrays for plotting
    t_np = t.numpy()
    y_np = audio.numpy()
    # # # Plot the sine wave
    plt.plot(t_np, y_np)
    plt.ylim(-1,1)

    plt.title(name)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig(f"{root}{name}")
    plt.clf()

def draw_heatmap(data, vmin=0, vmax=1, color_map='viridis',name="Test",root="./fig/"):
    data = np.array(data)
    
    plt.figure(figsize=(10, 8))
    
    heatmap = plt.imshow(data,vmin=vmin, vmax=vmax, cmap=color_map, aspect='auto')
    
    # cbar = plt.colorbar(heatmap)
    # cbar.ax.set_ylabel('Intensity')
    
    # Remove axis labels and ticks
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)



    # plt.title(name)

    plt.savefig(f"{root}{name}", bbox_inches='tight', pad_inches=0)
    
    # Display the heatmap
    plt.clf()

from PIL import Image

def gif_image(image_list,name,root="./fig/",duration=500):
    # Open the images
    images = [Image.open(image) for image in image_list]
    
    # Save as GIF
    images[0].save(
        f"{root}{name}.gif",
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

import json
def phone_to_phone_idx(pinyin:list):
    with open("./save/cache/phoneme.json","r") as f: 
        data = f.read()
        data = json.loads(data)
    phoidxdict = data["phoneme"]
    phoidxdict = {p:i for i, p in enumerate(phoidxdict)}
    with open("./save/cache/pinyin.json","r") as f:
        data = f.read()
        pinyin_dict = json.loads(data)
    phone_list = []
    tone_list = []
    for py in pinyin:
        if py == "sil":
            tone = 0
        else: 
            tone = int(py[-1])
            py = py[:-1]
        phone = pinyin_dict[py]
        tone = [0]*(len(phone)-1) + [tone]
        phone_list.append(phone)
        tone_list.append(tone)

    phone = [item for sublist in phone_list for item in sublist]
    tone = [item for sublist in tone_list for item in sublist]
    phone = [i if not i[-1].isdigit() else i[:-1] for i in phone]
    phone_idx = [phoidxdict[p]+1 for p in phone]
    return phone_idx, tone


def hanzi_to_pinyin(hanzi_string):
    # Convert Hanzi to Pinyin
    pinyin_list = pinyin(hanzi_string, style=Style.TONE3)
    # Join the pinyin with spaces
    output = [item[0] if item[0][-1].isdigit() else item[0]+"5" for item in pinyin_list]
    return ["sil"]*1+output+["sil"]*1

def hidden_to_audio(hidden):
    #hidden: B,T,C
    from params import params
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    from model import AE
    ae = AE(params).to(device)
    ae = loadModel(ae,"pqmf_ae","./checkpoint")
    hidden = torch.transpose(hidden,1,2)[0,:,:].unsqueeze(0)
    audio = ae.decode(hidden)
    audio = ae.pqmf.inverse(audio)
    audio = torch.clamp(audio, -1.0, 1.0)
    return audio