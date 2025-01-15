import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchaudio
import wandb
from sklearn.model_selection import train_test_split
from urbansounddataset import ESC50Dataset
from cnn import CNNNetwork

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10000
LEARNING_RATE = 0.001
SAMPLE_RATE = 16000
NUM_SAMPLES = 16000

# Dataset Parameters
ANNOTATIONS_FILE = "C:/Users/Codeexia/FinalSemester/Thesis/practice/AI_project/ESC-50-master/meta/esc50.csv"
AUDIO_DIR = "C:/Users/Codeexia/FinalSemester/Thesis/practice/AI_project/ESC-50-master/audio"
SELECTED_CLASSES = ["airplane", "helicopter", "train", "street_music", "engine", "siren", "car_horn"]

# Initialize wandb
wandb.init(project="esc50_urban_classification")
wandb.config.update({
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "sample_rate": SAMPLE_RATE,
    "num_samples": NUM_SAMPLES,
    "classes": SELECTED_CLASSES,
})

def create_data_loader(X, y, batch_size):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        wandb.log({"training_loss": loss.item()})

def validate_single_epoch(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    wandb.log({"validation_loss": avg_loss})
    return avg_loss

def train_and_validate(model, train_loader, val_loader, loss_fn, optimiser, device, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_single_epoch(model, train_loader, loss_fn, optimiser, device)
        avg_val_loss = validate_single_epoch(model, val_loader, loss_fn, device)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print("---------------------------")
    print("Training complete.")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # Mel Spectrogram transformation
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
        device=device,
        selected_classes=SELECTED_CLASSES,
    )

    # Convert dataset to features (X) and labels (y)
    X, y = [], []
    for signal, label in dataset:
        X.append(signal.cpu().numpy())  # Fix for GPU tensors
        y.append(label)  # Labels are scalar integers
    X = torch.stack([torch.tensor(x) for x in X]).numpy()
    y = torch.tensor(y).numpy()

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=41)

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Create data loaders
    train_loader = create_data_loader(X_train, y_train, BATCH_SIZE)
    val_loader = create_data_loader(X_val, y_val, BATCH_SIZE)

    # Model, loss function, optimizer
    model = CNNNetwork().to(device)  # No input_size or num_classes
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train and validate the model
    train_and_validate(model, train_loader, val_loader, loss_fn, optimizer, device, EPOCHS)

    # Save the model
    model_namefile = "esc50_urban_model4.pth"
    torch.save(model.state_dict(), model_namefile)
    print("Model saved as {model_namefile}")
