from scipy import fft
import numpy as np
from matplotlib import pyplot as plt

class FourierRegressor():
    
    def __init__(self):
        self.freq = []
        self.orig_freq = []
        self.power = []
        self.orig_power = []
        self.phase = []
        self.mean = 0
        self.amp = 0
        self.n = 5
        self.low_freq_filter = 0
    
    def fit(self, t, y, n = 5, low_freq_filter = 0):
        self.n = n
        self.mean = y.mean()
        self.amp = 2 * y.std()
        self.low_freq_filter = low_freq_filter
        self.power, self.freq, self.phase = self.__get_fundamental_components(y, 1, n)
        
    def predict(self, t):
        r = np.zeros(len(t))
        total_power = np.sum(self.power)
        for i in range(self.n):
            r += self.amp * (self.power[i] / total_power) * np.cos(2 * np.pi * self.freq[i] * t  + self.phase[i])
        return r + self.mean
    
    def plot_spectrogram(self, figsize=(20,10)):
        plt.figure(figsize=figsize)
        plt.plot(self.orig_freq, self.orig_power)
    
    def __get_fundamental_components(self, x, T, n):
        N = len(x)
        coef = fft.fft(x)[:N//2]
        freq = fft.fftfreq(N, T)[:N//2]
        
        self.orig_freq = freq
        self.orig_power = np.abs(coef)
        
        power = np.abs(coef)[freq > self.low_freq_filter]
        phase = np.angle(coef)[freq > self.low_freq_filter]
        freq = freq[freq > self.low_freq_filter]
            
        max_power_idx = np.argsort(power)[::-1][:n]
        return (power[max_power_idx], freq[max_power_idx], phase[max_power_idx])