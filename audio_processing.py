import torch
import torch.nn.functional as F
import numpy as np
import librosa
import warnings

class STFT(torch.nn.Module):
    def __init__(self, filter_length, hop_length, win_length, window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            fft_window = librosa.filters.get_window(window, win_length, fftbins=True)
            fft_window = librosa.util.pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        # Ensure input is 3D: (batch, channels, samples)
        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(0).unsqueeze(0)
        elif input_data.dim() == 2:
            input_data = input_data.unsqueeze(1)
        
        num_batches, num_channels, num_samples = input_data.size()
        
        # Reflect padding
        input_data = F.pad(
            input_data,
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        
        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part, real_part)

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = librosa.filters.window_sumsquare(
                                    window=self.window,
                                    n_frames=magnitude.size(-1),
                                    hop_length=self.hop_length,
                                    win_length=self.win_length,
                                    n_fft=self.filter_length,
                                    dtype=np.float32
                                )
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > np.finfo(np.float32).tiny)[0])
            window_sum = torch.from_numpy(window_sum).to(magnitude.device)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2)]

        return inverse_transform


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sample_rate=16000, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sample_rate = sample_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = torch.log(torch.clamp(magnitudes, min=1e-5))
        return output

    def spectral_de_normalize(self, magnitudes):
        return torch.exp(magnitudes)

    def mel_spectrogram(self, y):
        if torch.min(y.data) < -1 or torch.max(y.data) > 1:
                warnings.warn("Input audio contains values outside [-1, 1] range")

        magnitudes, phases = self.stft_fn.transform(y)
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

    def mel_spectrogram_to_wave(self, mel_spectrogram, n_iter=60):
        mel_spectrogram = self.spectral_de_normalize(mel_spectrogram)
        spectrogram = torch.matmul(self.mel_basis.transpose(0, 1), mel_spectrogram)
        
        # Initialize random phase
        angles = torch.rand_like(spectrogram) * 2 * np.pi
        
        # Griffin-Lim
        for _ in range(n_iter):
            full = torch.polar(spectrogram, angles)
            inverse = self.stft_fn.inverse(spectrogram, angles)
            _, angles = self.stft_fn.transform(inverse)

        waveform = self.stft_fn.inverse(spectrogram, angles)
        return waveform.squeeze(1)