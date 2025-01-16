import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchaudio
import wandb
from sklearn.model_selection import train_test_split
from urbansounddataset import ESC50Dataset
from cnn import CNNNetwork

# Hyperparameters
BATCH_SIZE = 32  # Number of samples per batch
EPOCHS = 500  # Number of epochs for training
LEARNING_RATE = 0.001  # Learning rate for optimizer
SAMPLE_RATE = 16000  # Audio sample rate
NUM_SAMPLES = 80000  # Number of audio samples per file ( )

# Dataset Parameters
ANNOTATIONS_FILE = "C:/Users/Codeexia/FinalSemester/Thesis/practice/AI_project/ESC-50-master/meta/esc50.csv"
AUDIO_DIR = "C:/Users/Codeexia/FinalSemester/Thesis/practice/AI_project/ESC-50-master/audio"
SELECTED_CLASSES = ["airplane", "helicopter", "train", "street_music", "engine", "siren", "car_horn"]

# Initialize wandb for experiment tracking
wandb.init(project="esc50_urban_classification")
wandb.config.update({
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "sample_rate": SAMPLE_RATE,
    "num_samples": NUM_SAMPLES,
    "classes": SELECTED_CLASSES,
})

def plot_learning_curves(train_losses, val_losses):
    # Plot the training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Curves')
    plt.show()

def create_data_loader(X, y, batch_size):
    # Create a DataLoader for batch processing
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    # Train the model for a single epoch
    model.train()
    epoch_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item()
        wandb.log({"training_loss": loss.item()})  # Log training loss
    return epoch_loss / len(data_loader)

def validate_single_epoch(model, data_loader, loss_fn, device):
    # Evaluate the model on the validation set
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    wandb.log({"validation_loss": avg_loss})  # Log validation loss
    return avg_loss

def train_and_validate(model, train_loader, val_loader, loss_fn, optimiser, device, epochs):
    # Train and validate the model over multiple epochs
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_single_epoch(model, train_loader, loss_fn, optimiser, device)
        val_loss = validate_single_epoch(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print("---------------------------")

    plot_learning_curves(train_losses, val_losses)
    print("Training complete.")

if __name__ == "__main__":
    # Set the device for computation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    # Mel Spectrogram transformation for audio preprocessing
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
    )
    normalize = torchaudio.transforms.AmplitudeToDB(top_db=80)  # Normalize amplitude to decibels between 0 and 80 dB
    transformation = nn.Sequential(mel_spectrogram, normalize) # Combine transformations

    # Load and preprocess the dataset
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
        X.append(signal.cpu().numpy())  # Convert tensors to numpy arrays
        y.append(label)  # Collect labels
    X = torch.stack([torch.tensor(x) for x in X]).numpy()
    y = torch.tensor(y).numpy()

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=41)

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Create DataLoaders for batch processing
    train_loader = create_data_loader(X_train, y_train, BATCH_SIZE)
    val_loader = create_data_loader(X_val, y_val, BATCH_SIZE)

    # Initialize model, loss function, and optimizer
    model = CNNNetwork().to(device)  # Load CNN architecture
    loss_fn = nn.CrossEntropyLoss()  # Define loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Set optimizer

    # Train and validate the model
    train_and_validate(model, train_loader, val_loader, loss_fn, optimizer, device, EPOCHS)

    # Save the trained model to a file
    model_namefile = "esc50_urban_model4.pth"
    torch.save(model.state_dict(), model_namefile)
    print(f"Model saved as {model_namefile}")
