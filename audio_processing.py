import torch
import torchaudio
from omegaconf import OmegaConf
import numpy as np
import torchaudio.transforms as T

config = OmegaConf.load("config.yaml")
config.n_stft = int(config.n_fft // 2 + 1)
config.hop_length = int(config.n_fft / 8.0)
config.win_length = int(config.n_fft / 2.0)

spec_transform = torchaudio.transforms.Spectrogram(
    n_fft=config.n_fft, 
    win_length=config.win_length,
    hop_length=config.hop_length,
    power=config.power
)

mel_scale_transform = torchaudio.transforms.MelScale(
  n_mels=config.n_mel_channels, 
  sample_rate=config.sample_rate, 
  n_stft=config.n_stft
)

mel_inverse_transform = torchaudio.transforms.InverseMelScale(
    n_mels=config.n_mel_channels, 
    sample_rate=config.sample_rate, 
    n_stft=config.n_stft
)

def pre_emphasis(x, coef=0.97):
    return torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - coef * x[:, :-1]), dim=1)

def de_emphasis(x, coef=0.97):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x_ = x.clone()
    for i in range(1, x_.size(-1)):
        x_[..., i] += coef * x_[..., i-1]
    return x_.squeeze(0)

def norm_mel_spec_db(mel_spec):  
    mel_spec = ((2.0*mel_spec - config.min_level_db) / (config.max_db/config.norm_db)) - 1.0
    mel_spec = torch.clip(mel_spec, -config.ref*config.norm_db, config.ref*config.norm_db)
    return mel_spec

def denorm_mel_spec_db(mel_spec):
    mel_spec = (((1.0 + mel_spec) * (config.max_db/config.norm_db)) + config.min_level_db) / 2.0 
    return mel_spec

def ensure_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x

def db_to_power_mel_spec(mel_spec):
    mel_spec = ensure_tensor(mel_spec)
    mel_spec = mel_spec * config.scale_db
    mel_spec = torchaudio.functional.DB_to_amplitude(
        mel_spec,
        ref=config.ampl_ref,
        power=config.ampl_power
    )  
    return mel_spec

def inverse_mel_spec_to_wav(mel_spec, n_iter=60):
    mel_spec = ensure_tensor(mel_spec)
    power_mel_spec = db_to_power_mel_spec(mel_spec).cpu()
    spectrogram = mel_inverse_transform(power_mel_spec)
    
    griffnlim_transform = torchaudio.transforms.GriffinLim(
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        n_iter=n_iter
    )
    
    pseudo_wav = griffnlim_transform(spectrogram)
    
    return pseudo_wav

def pow_to_db_mel_spec(mel_spec):
    mel_spec = ensure_tensor(mel_spec)
    mel_spec = torchaudio.functional.amplitude_to_DB(
        mel_spec,
        multiplier = config.ampl_multiplier, 
        amin = config.ampl_amin, 
        db_multiplier = config.db_multiplier, 
        top_db = config.max_db
    )
    mel_spec = mel_spec/config.scale_db
    return mel_spec

def loudness_normalize(wav, target_lufs=-23.0, sample_rate=16000):
    meter = T.Loudness(sample_rate)
    
    # Ensure wav is 2D: (channels, samples)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.dim() > 2:
        raise ValueError(f"Expected 1D or 2D tensor, got {wav.dim()}D")
    
    # Ensure proper shape: (batch, channels, samples) for meter
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)
    
    loudness = meter(wav)
    gain = 10**((target_lufs - loudness) / 20)
    
    # Apply gain and remove batch dimension if it was added
    normalized_wav = (wav.squeeze(0) * gain).clamp(-1, 1)
    
    return normalized_wav

def convert_to_mel_spec(wav):
    wav = ensure_tensor(wav)
    
    # Apply loudness normalization
    wav = loudness_normalize(wav, target_lufs=-23.0, sample_rate=config.sample_rate)
    
    # Apply pre-emphasis
    wav = pre_emphasis(wav)
    
    spec = spec_transform(wav)
    mel_spec = mel_scale_transform(spec)
    db_mel_spec = pow_to_db_mel_spec(mel_spec)
    db_mel_spec = db_mel_spec.squeeze(0)
    if db_mel_spec.shape[-1] > 800:
        db_mel_spec = db_mel_spec[:, :800]
    return db_mel_spec