import numpy as np
import pandas as pd
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1))
    return mask


def load_wav_to_torch(full_path):
    sample_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sample_rate


def load_filepaths_and_text(filename):
    file_paths_and_text = pd.read_csv(filename)
    return file_paths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x