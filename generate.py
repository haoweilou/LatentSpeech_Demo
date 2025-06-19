from tts import StyleSpeech
from model import DiffusionBlock
import torch
from tts_config import config
from function import loadModel,save_audio, hidden_to_audio, phone_to_phone_idx,hanzi_to_pinyin
from params import params
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tts_model = StyleSpeech(config).to(device)
tts_model = loadModel(tts_model,f"tts","./checkpoint")

diffusion_block = DiffusionBlock(params).to(device)
diffusion_block = loadModel(diffusion_block,f"diff","./checkpoint")

tts_model.eval()
diffusion_block.eval()

beta = np.array(params.noise_schedule)
noise_level = np.cumprod(1 - beta)
noise_level = torch.tensor(noise_level.astype(np.float32))

def noise_schedule(tts_embed,training_noise_schedule,inference_noise_schedule):
    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    T = []
    for s in range(len(inference_noise_schedule)):
        for t in range(len(training_noise_schedule) - 1):
            if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                T.append(t + twiddle)
                break
    T = np.array(T, dtype=np.float32)
    tts_embed = tts_embed.to(device)#B,T,C
    hidden = torch.rand_like(tts_embed, device=device)
    hidden = torch.transpose(hidden,1,2) #B,C,T
    with torch.no_grad():
        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n]**0.5
            c2 = beta[n] / (1 - alpha_cum[n])**0.5
            predict_noise = diffusion_block(hidden,torch.tensor([T[n]]).to(device),tts_embed) #B,C,T
            mean = torch.mean(predict_noise).item()
            std = torch.std(predict_noise).item()
            print(f"Diffusion Step: {n}, Mean: {mean:.3f}, STD: {std:.3f}")
            hidden = c1 *(hidden - c2 * predict_noise)
            if n > 0:
                noise = torch.randn_like(hidden)
                sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                hidden += sigma * noise
    return hidden

training_noise_schedule = np.array(params.noise_schedule)
inference_noise_schedule = np.array(params.noise_schedule)

hanzi = "你好很高兴认识你"
pinyin = hanzi_to_pinyin(hanzi)
phone_idx,tone = phone_to_phone_idx(pinyin)
pho_len = len(phone_idx)
print("Pinyin: "," ".join(pinyin))
print("Pinyin Idx: ",phone_idx,"\n","Pinyin Tone: ",tone)
d = 20
duration = torch.tensor([[d for _ in range(len(phone_idx))]]).to(device)
phone_mask = torch.tensor([[0 for _ in range(len(phone_idx))]]).to(device)
phone_idx = torch.tensor([phone_idx]).to(device)
tone = torch.tensor([tone]).to(device)
hidden_mask = torch.tensor([[0 for _ in range(1024)]]).to(device)
src_lens = torch.tensor([phone_idx.shape[-1]]).to(device)
mel_lens = torch.tensor([d*phone_idx.shape[-1]]).to(device)
import time
start_time = time.time()
tts_embed,log_l_pred,mel_masks = tts_model(phone_idx,tone,src_lens=src_lens,mel_lens=mel_lens,max_mel_len=config["max_seq_len"])
end_time = time.time()
print(f"TTS Execution time: {end_time-start_time} seconds")

durations = torch.round(torch.exp(log_l_pred) - 1)
duration_value = int(torch.sum(durations.sum()).item())
tts_embed = tts_embed[:,:duration_value,:]
start_time = time.time()
hidden = noise_schedule(tts_embed,training_noise_schedule,inference_noise_schedule)
end_time = time.time()
print(f"Denoiser time: {end_time-start_time:.2f} seconds")

start_time = time.time()
audio = hidden_to_audio(torch.transpose(hidden,1,2)).detach().cpu()[0]
end_time = time.time()
print(f"AE Decider time: {end_time-start_time:.2f} seconds")
save_audio(audio,48000,f"output","./")