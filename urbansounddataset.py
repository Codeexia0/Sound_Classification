import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from torch import nn

class ESC50Dataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device, selected_classes):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

        # Filter annotations for selected classes
        # This filters the annotations DataFrame to include only rows where the "category"
        # column contains one of the `selected_classes`. The index is reset to ensure a clean
        # sequential order without gaps from the filtered DataFrame.
        self.annotations = self.annotations[self.annotations["category"].isin(selected_classes)].reset_index(drop=True)

        # Map class labels to indices
        # Create a mapping of class names to unique numerical indices. This is essential for converting
        # the class labels (e.g., "airplane", "helicopter") into numerical values required for the
        # machine learning model. The mapping is stored in the `self.class_mapping` dictionary.
        self.class_mapping = {category: idx for idx, category in enumerate(selected_classes)}


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        filename = self.annotations.iloc[index, 0]
        return os.path.join(self.audio_dir, filename)

    def _get_audio_sample_label(self, index):
        category = self.annotations.iloc[index, 3]
        return self.class_mapping[category]


if __name__ == "__main__":
    # Parameters
    ANNOTATIONS_FILE = "C:/Users/Codeexia/FinalSemester/Thesis/practice/AI_project/ESC-50-master/meta/esc50.csv"
    AUDIO_DIR = "C:/Users/Codeexia/FinalSemester/Thesis/practice/AI_project/ESC-50-master/audio"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 16000
    SELECTED_CLASSES = ["airplane", "helicopter", "train", "street_music", "engine", "siren", "car_horn"]

    # Transformation
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    normalize = torchaudio.transforms.AmplitudeToDB(top_db=80)
    transformation = nn.Sequential(mel_spectrogram, normalize)

    # Load dataset
    dataset = ESC50Dataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=mel_spectrogram,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        device="cpu",
        selected_classes=SELECTED_CLASSES,
    )

    print(f"Loaded {len(dataset)} samples.")
    signal, label = dataset[0]
    print(f"Signal shape: {signal.shape}, Label: {label}")
