import json
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import shuffle as shuffle

with open('nsynth-test/data.json') as f:
    data = json.load(f)

# print(data["bass_synthetic_068-049-025"]['pitch'])
# for i in data.keys():
#     y = librosa.load('nsynth-test/audio/' + i + ".wav")[0]

y, sr = librosa.load('nsynth-test/audio/bass_synthetic_068-108-050.wav')
t = librosa.load('nsynth-test/audio/bass_electronic_018-052-075.wav')[0]
x = librosa.load('nsynth-test/audio/bass_electronic_018-063-075.wav')[0]
# librosa.output.write_wav('short.wav', y[:sr*2], sr)
# librosa.output.write_wav('normal.wav', y, sr)

# for i in range(1,11):
#     librosa.output.write_wav('short'+str(i)+'.wav', y[:int(sr/i)], sr)

def spectrogram(y):
    S = np.abs(librosa.stft(y))
    return S
def show_spectrogram(y, sr):
    S = spectrogram(y)
    import librosa.display as display
    display.specshow(librosa.power_to_db(S**2, ref=np.max), sr=sr, y_axis = 'log' , x_axis='time')
    plt.tight_layout()
    plt.show()

# for i in spectrogram(y)[:,100]:
#     print(i)
#print(spectrogram(y)[:,0])

def show_cqt(y,sr,bins_per_octave):
    C = np.abs(librosa.cqt(y,sr=sr, fmin = librosa.midi_to_hz(0), n_bins = 5*bins_per_octave, bins_per_octave = bins_per_octave))
    import librosa.display as display
    display.specshow(librosa.amplitude_to_db(C,ref = np.max), sr=sr, x_axis = 'time', y_axis = 'cqt_note', bins_per_octave = bins_per_octave, fmin = librosa.midi_to_hz(0), fmax = librosa.midi_to_hz(127))
    plt.colorbar(format = '%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.tight_layout()
    plt.show()

# bass32real = librosa.load('nsynth-test/bass_electronic_018-032-127.wav')
# bass32fake = librosa.load('nsynth-test/audio/bass_electronic_018-044-075.wav')
# show_cqt(bass32real[0], bass32real[1], 24)
# show_cqt(bass32fake[0], int(bass32fake[1]/2), 24)

#checking the maximum and minimum pitches in our dataset
# maxPitch=0
# minPitch=127
# for i in data.keys():
#     pitch = data[i]["pitch"]
#     if pitch<minPitch:
#         minPitch=pitch
#     if pitch > maxPitch:
#         maxPitch = pitch
# print(minPitch, maxPitch)


count = 0
database = []
BINS_PER_OCTAVE = 24
NUM_OF_OCVTAVES = 8
for i in data.keys():
    pitch = data[i]["pitch"]
    if  (pitch < 85 and pitch > 24):
        y, sr = librosa.load('nsynth-test/audio/'+ i + '.wav')
        C = np.abs(librosa.cqt(y,sr=sr, fmin = librosa.midi_to_hz(20), n_bins = NUM_OF_OCVTAVES*BINS_PER_OCTAVE, bins_per_octave = BINS_PER_OCTAVE))
        for row in np.split(C,C.shape[1], axis=1)[0:40]:
            database.append((row, data[i]["instrument_family"]))
        if count % 100 == 0:
            print(count)
        count +=1

shuffle(database)

training_data = database[:int(0.8*len(database))]
val_data = database[int(0.8*len(database)):int(0.9*len(database))]
test_data = database[int(0.9*len(database)):]



with open('newdataset.pickle',"wb") as f:
    pickle.dump([training_data,val_data,test_data],f)
