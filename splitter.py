import wave
import scipy.io.wavfile as w
import matplotlib.pyplot as plt
import librosa
import numpy as np
from os import listdir, path

data = [f for f in listdir('nsynth-test/audio')]
inputs = [librosa.load(f.path) for f in data]
print(inputs)

x = wave.open("cartoon001.wav",mode='rb')
print(x.getnchannels(),x.getframerate(),x.getnframes())
x.readframes(10000)
fs,y = w.read('cartoon001.wav')
print(y.size)
print(fs)

#returns the number of seconds of the audio file
def seconds(sample, samplerate):
    return sample.size/samplerate

#plots the audio file with respect to seconds
def plot_sec(sample, samplerate):
    t = np.arange(sample.size)/float(samplerate)
    plt.plot(t,sample)
    plt.show()

#plots the audio file with respect to sample size
def plot_normal(sample):
    plt.plot(sample)
    plt.show()

x=y[0:20000]
w.write('short.wav', fs, x)

z=np.floor_divide(y,1000)
w.write('quite.wav', fs, z)

x,sr = librosa.load('cartoon001.wav')
S = np.abs(librosa.stft(x))
print(S)
import librosa.display as display
display.specshow(librosa.power_to_db(S**2, ref=np.max), sr=sr, y_axis = 'log', x_axis='time')
plt.tight_layout()
plt.show()
