import torch
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
from cnn import CNNNetwork
from urbansounddataset import ESC50Dataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES
import torch.nn as nn

class_mapping = [
    "airplane",
    "helicopter",
    "train",
    "street_music",
    "engine",
    "siren",
    "car_horn"
]

def predict(model, input, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0).item()
        predicted = class_mapping[predicted_index]
    return predicted, predicted_index

def plot_class_distribution(predictions, class_mapping):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=predictions, order=range(len(class_mapping)), palette="viridis") # Countplot for class distribution
    plt.xticks(range(len(class_mapping)), class_mapping, rotation=45)
    plt.title("Class-Wise Prediction Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.show()

def plot_confusion_matrix(true_labels, predicted_labels, class_mapping):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_mapping))) # Compute confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_mapping, yticklabels=class_mapping, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()

def visualize_sample_spectrogram(dataset, sample_index):
    sample, _ = dataset[sample_index]
    spectrogram = sample.squeeze().numpy() # Remove batch dimension
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram of Sample {sample_index}")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bands")
    plt.show()

if __name__ == "__main__":
    # Load the trained model
    cnn = CNNNetwork()  # Ensure the architecture matches the trained model
    state_dict = torch.load("esc50_urban_model4.pth", map_location="cpu")
    cnn.load_state_dict(state_dict)

    # Define the Mel Spectrogram transformation
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    normalize = torchaudio.transforms.AmplitudeToDB(top_db=80)
    transformation = nn.Sequential(mel_spectrogram, normalize)

    # Load the dataset
    usd = ESC50Dataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        "cpu",
        class_mapping
    )

    # Test on the entire dataset
    true_labels = []
    predicted_labels = []
    predictions = []

    for i in range(len(usd)):
        input, target = usd[i]
        input.unsqueeze_(0)  # Add batch dimension
        predicted, predicted_index = predict(cnn, input, class_mapping)

        true_labels.append(target)
        predicted_labels.append(predicted_index)
        predictions.append(predicted_index)

        expected = class_mapping[target]
        print(f"Sample {i + 1}/{len(usd)} - Predicted: '{predicted}', Expected: '{expected}'")

    # Visualization
    plot_class_distribution(predictions, class_mapping)
    plot_confusion_matrix(true_labels, predicted_labels, class_mapping)

    # Visualize spectrogram for a specific sample
    sample_index = 10  # Change as needed
    visualize_sample_spectrogram(usd, sample_index)
