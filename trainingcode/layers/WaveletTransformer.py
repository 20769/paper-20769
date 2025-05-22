import pywt
import numpy as np
import scipy.signal
import torch

class WaveletTransform:
    def __init__(self, fixed_max_len, wavelet='db1', level=None, mode='symmetric'):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.fixed_max_len = fixed_max_len

    def calculate_spectral_level(self, signal, sampling_rate):

        frequencies, psd = scipy.signal.welch(signal, fs=sampling_rate)
        dominant_freq = frequencies[np.argmax(psd)]
        level = int(np.log2(sampling_rate / (2 * dominant_freq)))
        
        return max(1, min(level, pywt.dwt_max_level(len(signal), pywt.Wavelet(self.wavelet).dec_len)))

    def select_wavelet(self, signal):

        if isinstance(signal, torch.Tensor):
            signal = signal.cpu().numpy()
        
        wavelets = pywt.wavelist(kind='discrete')
        best_wavelet = wavelets[0]
        best_score = float('inf')
        
        for wavelet in wavelets:
            coeffs = pywt.wavedec(signal, wavelet, level=1, mode=self.mode)
            score = np.sum(np.abs(coeffs[-1]))
            if score < best_score:
                best_score = score
                best_wavelet = wavelet
        
        return best_wavelet

    def transform(self, data, sampling_rate=None, max_len=None):
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
    
        bs, t, n = data.shape
    
        if self.wavelet == 'auto':
            self.wavelet = self.select_wavelet(data[0, :, 0])
    
        if self.level is None:
            if sampling_rate is not None:
                self.level = self.calculate_spectral_level(data[0, :, 0], sampling_rate)
            else:
                self.level = pywt.dwt_max_level(t, pywt.Wavelet(self.wavelet).dec_len)
    
        low_freq_len = len(pywt.wavedec(data[0, :, 0], self.wavelet, level=self.level, mode=self.mode)[0])
        result = np.zeros((bs, low_freq_len, n))
    
        for j in range(n):
            for i in range(bs):
                coeffs = pywt.wavedec(data[i, :, j], self.wavelet, level=self.level, mode=self.mode)
                result[i, :, j] = coeffs[0]
    
        if low_freq_len > max_len:
            new_result = np.zeros((bs, max_len, n))
            for j in range(n):
                for i in range(bs):
                    new_result[i, :, j] = np.interp(
                        np.linspace(0, low_freq_len - 1, max_len),
                        np.arange(low_freq_len),
                        result[i, :, j]
                    )
            result = new_result
        elif low_freq_len < max_len:
            new_result = np.zeros((bs, max_len, n))
            new_result[:, :low_freq_len, :] = result
            result = new_result
        else:
            result = result[:, :max_len, :]
        return result

def test_wavelet_transform():

    data = np.random.randn(2, 86, 3) 
    
    wt = WaveletTransform(wavelet='auto', level=None, mode='symmetric')

    transformed_data = wt.transform(data, sampling_rate=100)
    print("Transformed Data Shape:", transformed_data.shape)
