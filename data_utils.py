import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import SinhalaTokenizerTacotron
from audio_processing import TacotronSTFT


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, char_mapper, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_column_name = hparams.text_column_name
        self.audio_path_column_name = hparams.audio_path_column_name
        self.max_wav_value = hparams.max_wav_value
        self.sample_rate = hparams.sample_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sample_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.text_to_sequence = self._load_tokenizer(char_mapper)
        random.seed(hparams.seed)
        self.audiopaths_and_text = self.audiopaths_and_text.sample(frac=1, ignore_index=True)


    def _load_tokenizer(self, char_mapper):
        _tokenizer = SinhalaTokenizerTacotron(text_list=None, vocab_map=char_mapper)
        return _tokenizer

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[self.audio_path_column_name], audiopath_and_text[self.text_column_name]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        audio, sample_rate = load_wav_to_torch(filename)
        if sample_rate != self.stft.sample_rate:
            raise ValueError(f"File SR {sample_rate} doesn't match target SR {self.stft.sample_rate}")
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        return torch.squeeze(melspec, 0)

    def get_text(self, text):
        seq = self.text_to_sequence(text, truncate_and_pad=False)
        # print(seq)
        text_norm = torch.IntTensor(seq)
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text.iloc[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths
