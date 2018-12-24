import keras
import numpy as np
import librosa
import librosa.display as display
import json
import matplotlib.pyplot as plt

with open('nsynth-test/data.json') as f:
    data = json.load(f)

family_dict = {
            0: "bass",
            1: "brass",
            2: "flute",
            3: "guitar",
            4: "keyboard",
            5: "mallet",
            6: "organ",
            7: "reed",
            8: "string",
            9: "synth_lead",
            10: "vocal",
}

model = keras.models.load_model('simple_model.h5')

def show_cqt(y,sr,bins_per_octave):
    C = np.abs(librosa.cqt(y,sr=sr, fmin = librosa.midi_to_hz(20), n_bins = 8*bins_per_octave, bins_per_octave = bins_per_octave, filter_scale = 2))
    display.specshow(librosa.amplitude_to_db(C,ref = np.max), sr=sr, x_axis = 'time', y_axis = 'cqt_note', bins_per_octave = bins_per_octave, fmin = librosa.midi_to_hz(0), fmax = librosa.midi_to_hz(127))
    plt.colorbar(format = '%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.tight_layout()
    plt.show()


BINS_PER_OCTAVE = 12
NUM_OF_OCVTAVES = 8
for i in data.keys():
    pitch = data[i]["pitch"]
    if pitch<80:
        y, sr = librosa.load('nsynth-test/audio/'+ i + '.wav')
        print(sr)
        C = np.abs(librosa.cqt(y,sr=sr, fmin = librosa.midi_to_hz(20), n_bins = NUM_OF_OCVTAVES*BINS_PER_OCTAVE, bins_per_octave = BINS_PER_OCTAVE, pad_mode='reflect'))
        prediction = np.argmax(model.predict(C.reshape(1,C.shape[0],C.shape[1],1)))
        if prediction!=data[i]["instrument_family"]:
            print("real: "+data[i]["instrument_family_str"] + ", prediction: " +family_dict[prediction] + " -- pitch: "+str(pitch) +" -- key: "+i)
            show_cqt(y,sr,2*BINS_PER_OCTAVE)
        else:
            print("correct!  "+data[i]["instrument_family_str"]+" -- pitch: "+str(pitch) +" -- key: "+i)
            show_cqt(y,sr,2*BINS_PER_OCTAVE)
