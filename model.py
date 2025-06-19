import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Optional, Sequence
import warnings
from pqmf import PQMF
from math import sqrt

def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer

@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)

class DiffusionEmbedding(nn.Module):
    """Some Information about DiffusionEmbedding"""
    def __init__(self,max_steps):
        super(DiffusionEmbedding, self).__init__()
        self.register_buffer("embedding",self._build_embedding(max_steps),persistent=False)
        self.projection1 = nn.Linear(128, 512)
        self.projection2 = nn.Linear(512, 512)
        
    def forward(self,diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x
    
    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)
    def _build_embedding(self,max_steps):
        #sinusoidal positional embedding
        steps = torch.arange(max_steps).unsqueeze(1)    #[T,1]
        dims = torch.arange(64).unsqueeze(0)            #[64,1]
        table = steps * 10**(dims * 4 /63)              #[T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1) #[T,128]
        return table

class ResidualBlock(nn.Module):
    """Some Information about ResidualBlock"""
    def __init__(self, n_mels, residual_channels, dilation, uncond=False):
        super(ResidualBlock, self).__init__()
        self.diffusion_projection = nn.Linear(512, residual_channels)
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner):
        #X: [B, C, L], t: [B,512], conitioner: [B,N_MEL,L]
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1) #t: [B, C, 1]
        y = x + diffusion_step
        conditioner = self.conditioner_projection(conditioner) #[B, 2C, L]
        y = self.dilated_conv(y) + conditioner #[B,2C,L]+[B,2C,L]
        gate, filter = torch.chunk(y, 2, dim=1) #[B,C,L], [B,C,L]
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y) #[B,C,L] => [B,2C,L]
        residual, skip = torch.chunk(y, 2, dim=1) #[B,C,L], [B,C,L]

        return (x+residual) / sqrt(2.0), skip        

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super(ResidualLayer, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(num_hiddens, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        inputs = x
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        return x + inputs
    
warnings.filterwarnings("ignore")
class MultiScaleSTFT(nn.Module):

    def __init__(self,
                 scales: Sequence[int],
                 sample_rate: int,
                 magnitude: bool = True,
                 normalized: bool = False,
                 num_mels: Optional[int] = None) -> None:
        super().__init__()
        self.scales = scales
        self.magnitude = magnitude
        self.num_mels = num_mels

        self.stfts = []
        for scale in scales:
            self.stfts.append(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=scale,
                    win_length=scale,
                    hop_length=scale // 4,
                    normalized=normalized,
                    n_mels=num_mels,
                    power=1,
                ))

        self.stfts = nn.ModuleList(self.stfts)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        b,c,t = x.shape
        x = torch.reshape(x,(b*c,t))
        stfts = []
        for stft in self.stfts:
            y = stft(x)
            stfts.append(y.abs())

        return stfts
    
class AudioDistance(nn.Module):

    def __init__(self, params,
                 log_epsilon: float) -> None:
        super().__init__()
        self.multiscale_stft = MultiScaleSTFT(params.stft_scales,params.sample_rate,num_mels=params.n_mels)
        self.log_epsilon = log_epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        stfts_x = self.multiscale_stft(x)
        stfts_y = self.multiscale_stft(y)
        distance = 0.

        for x, y in zip(stfts_x, stfts_y):
            logx = torch.log(x + self.log_epsilon)
            logy = torch.log(y + self.log_epsilon)

            lin_distance = F.mse_loss(x,y)
            log_distance = F.l1_loss(logx, logy)

            distance = distance + lin_distance + log_distance

        return distance

 
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zero_(m.bias)

class AE(nn.Module):
    def __init__(self,params,hidden_dim=16):
        super(AE, self).__init__()
        self.params = params
        self.spec_distance = AudioDistance(params,params.log_epsilon)
        self.pqmf = PQMF(100,params.n_band)
        self.encoder = nn.Sequential(
            nn.Conv1d(params.n_band,params.n_band*4,7,padding=3),
            nn.BatchNorm1d(params.n_band*4),
            nn.LeakyReLU(.2),
            nn.Conv1d(params.n_band*4,params.n_band*8,kernel_size=4*2+1,stride=4,padding=4),
            nn.BatchNorm1d(params.n_band*8),
            nn.LeakyReLU(.2),
            nn.Conv1d(params.n_band*8,params.n_band*16,kernel_size=4*2+1,stride=4,padding=4),
            nn.BatchNorm1d(params.n_band*16),
            nn.LeakyReLU(.2),
            nn.Conv1d(params.n_band*16,params.n_band*32,kernel_size=4*2+1,stride=4,padding=4),
            nn.BatchNorm1d(params.n_band*32),
            nn.LeakyReLU(.2),
            nn.Conv1d(params.n_band*32,hidden_dim,1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(.2),
        )

        self.encoder.apply(weights_init)

        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim,params.n_band*32,1),
            nn.BatchNorm1d(params.n_band*32),
            nn.LeakyReLU(.2),
            nn.ConvTranspose1d(params.n_band*32,params.n_band*16,kernel_size=4*2,stride=4,padding=4//2),
            nn.BatchNorm1d(params.n_band*16),
            nn.LeakyReLU(.2),
            nn.ConvTranspose1d(params.n_band*16,params.n_band*8,kernel_size=4*2,stride=4,padding=4//2),
            nn.BatchNorm1d(params.n_band*8),
            nn.LeakyReLU(.2),
            nn.ConvTranspose1d(params.n_band*8,params.n_band*4,kernel_size=4*2,stride=4,padding=4//2),
            nn.BatchNorm1d(params.n_band*4),
            nn.LeakyReLU(.2),
        )
        self.decoder.apply(weights_init)
        self.wave_gen = nn.Conv1d(params.n_band*4,params.n_band,7,padding=3)
        self.loud_gen = nn.Conv1d(params.n_band*4,params.n_band,3,1,padding=1)
        
    def mod_sigmoid(self,x):
        return 2 * torch.sigmoid(x)**2.3 + 1e-7
    
    def decode(self,x):
        z_ = self.decoder(x)
        loud = self.loud_gen(z_)
        wave = self.wave_gen(z_)
        x = torch.tanh(wave) *  self.mod_sigmoid(loud)
        return x
    
    def encode(self,x):
        mb_auido = self.pqmf(x)
        z = self.encoder(mb_auido)
        return z,mb_auido
    
    def forward(self, x):
        z,mb_auido = self.encode(x)
        x_reconstruct = self.decode(z)

        b,c,t = mb_auido.shape
        x_reconstruct = x_reconstruct[:,:,:t]

        input_audio = torch.reshape(mb_auido,(b*c,1,t))
        output_audio = torch.reshape(x_reconstruct,(b*c,1,-1))
        distance = self.spec_distance(input_audio,output_audio)
        l2_distance = F.mse_loss(input_audio,output_audio)

        x_reconstruct = self.pqmf.inverse(x_reconstruct)
        return x_reconstruct,distance,l2_distance
    
class DiffusionBlock(nn.Module):
    def __init__(self,params,input_channel=16):
        super(DiffusionBlock, self).__init__()
        self.params = params
        #audio channel = 1, output channel = C, kenerl_size = 1
        self.input_channel = input_channel
        self.input_project = Conv1d(self.input_channel,params.residual_channels,1)
        #diffusion time embedding
        self.diffusion_embedding = DiffusionEmbedding(params.max_step)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(input_channel, params.residual_channels, 2**(i % params.dilation_cycle_length), uncond=params.unconditional)
            for i in range(params.residual_layers*8)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, self.input_channel, 1)

    def forward(self, y, diffusion_step, x=None):
        #Generate hidden feature y condition on x
        #B,C,T
        x = torch.transpose(x,1,2) #B,T,C => B,C,T
        #B, C, L => B, C, L
        y = self.input_project(y)
        #B, C, 512
        diff_embed = self.diffusion_embedding(diffusion_step)
        #B, C ,L
        skip = None
        i = 0
        for layer in self.residual_layers:
            i += 1
            y, skip_connection = layer(y, diff_embed, x)
            skip = skip_connection if skip is None else skip_connection + skip
        #B,C,L
        y = skip / sqrt(len(self.residual_layers))
        y = self.skip_projection(y)
        y = F.leaky_relu(y)
        #[B,C,L] => [B,C,L]
        y = self.output_projection(y)
        return y