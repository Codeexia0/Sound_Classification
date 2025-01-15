import torch
import torchaudio
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
    correct_predictions = 0
    total_samples = len(usd)

    for i in range(total_samples):
        input, target = usd[i]
        input.unsqueeze_(0)  # Add batch dimension
        predicted, predicted_index = predict(cnn, input, class_mapping)
        expected = class_mapping[target]

        # Check if prediction is correct
        is_correct = predicted_index == target
        result = "Correct" if is_correct else "Incorrect"

        if is_correct:
            correct_predictions += 1

        print(f"Sample {i + 1}/{total_samples} - Predicted: '{predicted}', Expected: '{expected}' - {result}")

    # Calculate and display accuracy
    accuracy = (correct_predictions / total_samples) * 100
    print(f"\nTotal Correct Predictions: {correct_predictions}/{total_samples}")
    print(f"Accuracy: {accuracy:.2f}%")
