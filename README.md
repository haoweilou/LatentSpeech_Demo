# LatentSpeech: Efficient Diffusion-Based Text-to-Speech in the PQMF Latent Space
<p float="left">
  <img src="./asset/overall.png" width="35%" alt="LatentSpeech Overview" />
  <img src="./asset/denoiser.png" width="55%" alt="LatentSpeech System Architecture" />
</p>


## Abstract
Text-to-Speech (TTS) generation plays a crucial role in human-robot interaction by allowing robots to communicate naturally with humans. Researchers have developed various TTS models to enhance speech generation. More recently, diffusion models have emerged as a powerful generative framework, achieving state-of-the-art performance in tasks such as image and video generation. However, their application in TTS has been limited by slow inference speeds due to the iterative denoising process. Previous work has applied diffusion models to Mel-spectrograms with an additional vocoder to convert them into waveforms.

To address these limitations, we propose **LatentSpeech**, a novel diffusion-based TTS framework that operates directly in a latent space. This space is significantly more compact and information-rich than raw Mel-spectrograms. Furthermore, we introduce an alternative latent space of **Pseudo-Quadrature Mirror Filters (PQMF)**, which decomposes speech into multiple subbands. By leveraging PQMF's near-perfect waveform reconstruction capability, LatentSpeech eliminates the need for a separate vocoder and reduces both model size and inference time.

Our PQMF-based LatentSpeech model reduces inference time by **45%** and model size by **77%** compared to Mel-spectrogram diffusion models. On benchmark datasets, it achieves **25% lower WER** and **58% higher MOS** using the same training data. These results highlight LatentSpeech as an efficient, high-quality TTS solution for real-time and human-robot interaction.

## Instructions
1. **Clone the repository**
```
git clone https://github.com/haoweilou/LatentSpeech_Demo.git
```

2. **Download the checkpoint**

Download the checkpoint from the link below and unzip it into the root folder:
[Checkpoint](https://unsw-my.sharepoint.com/:u:/g/personal/z5258575_ad_unsw_edu_au/EbJhvIBylPhFjI5kc8EOkksBLTIB32ecT7cfhcAzvk_hrw?e=4lSNev)

3. **Modify input text**

Open generate.py and change the text on line 58 to what you want to synthesize.

4. **Run the script**
```
python generate.py
```

The generated speech will be saved as ```output.wav```.

⚠️ Note: This project only supports Chinese text-to-speech generation.

Enjoy