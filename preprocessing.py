import numpy as np
import pandas as pd
import copy
from torch.utils.data import DataLoader, Dataset, random_split
import librosa
import IPython.display as ipd
import random
from glob import glob
from sklearn.model_selection import train_test_split
import torch



def get_mel_spectrogram(audio, sr=16000, n_mels=40, hop_length=512, n_fft=1024):
    # return the MEL spectrogram after converting it into Db
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram

def generate_random_samples(audio, label, sample_rate=16000, sample_seconds=3, n_samples=10, n_mels=40, n_fft=1024, hop_length=512):
    
    samples = []
    labels = []
    total_samples = int(sample_seconds * sample_rate)
    for sample in range(n_samples):
        start = random.randint(0, len(audio) - total_samples)
        end = start + total_samples
        mel_spec = get_mel_spectrogram(audio[start : end],
                                       n_mels=n_mels,
                                       hop_length=hop_length,
                                       n_fft=n_fft,
                                       sr=sample_rate)
        # normalising
        mel_spec = np.abs(mel_spec)
        mel_spec /= 80
        samples.append(mel_spec)
        labels.append([label])
    
    
    return np.array(samples), np.array(labels)
        
        

#read the dataframe
df = pd.read_csv("./archive/KAGGLE/DATASET-balanced.csv")


#laod the audio files
real_audio = glob("./archive/KAGGLE/AUDIO/REAL/*")
fake_audio = glob("./archive/KAGGLE/AUDIO/FAKE/*")

combined_samples = []
combined_labels = []

HOP_LENGTH = 512
SAMPLE_RATE = 22050
N_MELS = 40
N_FFT = 1024

for path in real_audio:
    audio, sr = librosa.load(path)
    # 0 for real, 1 for fake
    s, l = generate_random_samples(audio, 0, sample_rate=SAMPLE_RATE, sample_seconds=1.5, n_samples=100, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    combined_samples.append(s)
    combined_labels.append(l)

for path in random.choices(fake_audio, k=8):
    audio, sr = librosa.load(path)
    # 0 for real, 1 for fake
    s, l = generate_random_samples(audio, 1, sample_rate=SAMPLE_RATE, sample_seconds=1.5, n_samples=100, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    combined_samples.append(s)
    combined_labels.append(l)
    

#get combined samples
combined_samples = np.concatenate(combined_samples, axis=0)
combined_labels = np.concatenate(combined_labels, axis=0)

class CustomDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample = sample[np.newaxis, :, :]
        # sample = sample.repeat(3, axis=0)
        return sample, np.array(self.labels[idx])
    

#Create the dataset
complete_dataset = CustomDataset(combined_samples, combined_labels)

#Split the data
train_size = int(0.8*len(complete_dataset))
test_size = int(0.1*len(complete_dataset))
validation_size = int(0.1*len(complete_dataset))
train_data, test_data, validation_data = random_split(complete_dataset, [train_size, test_size, validation_size])

# X, X_test, y, y_test = train_test_split(combined_samples, combined_labels, train_size=0.9, test_size=0.1, shuffle=True)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
# train_dataset = CustomDataset(X_train, y_train)
# test_dataset = CustomDataset(X_test, y_test)


train_dataloader = DataLoader(train_data, batch_size=16)
test_dataloader = DataLoader(test_data, batch_size=16)
validation_data = DataLoader(validation_data, batch_size=16)

# for data in train_dataloader:
#     print(data)
#     break