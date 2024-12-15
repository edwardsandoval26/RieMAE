import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent)) #
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Import tqdm for progress bars


def run_model(model, dataloader, criterion, device, optimizer=None):
    """
    Generalized function for training or evaluating the model.

    Parameters:
    - model: PyTorch model to train or evaluate.
    - dataloader: PyTorch DataLoader for the dataset.
    - criterion: Loss function (e.g., nn.MSELoss).
    - device: Device to use ('cuda' or 'cpu').
    - optimizer: Optimizer for training. If None, the model is evaluated.

    Returns:
    - avg_loss: Average loss over the dataset.
    """
    if optimizer:
        model.train()  # Training mode
    else:
        model.eval()  # Evaluation mode

    running_loss = 0.0

    with torch.set_grad_enabled(optimizer is not None):  # Enable gradients only if training
        i = 0
        for batch in dataloader:
            #Print batch shape
            #print(f"Batch shape: {batch.shape}")
            s_shape = batch.shape
            batch = batch.view(-1, 1, s_shape[-1], s_shape[-1])
            #Squeeze the 0 axis and then add a new axis in index 1
            
            inputs = batch.to(device)

            # Forward pass
            #print(f"Doing batch #{i}")
            i += 1
            outputs = model(inputs)

            loss = criterion(outputs[0], inputs)

            if optimizer:
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Accumulate loss
            running_loss += loss.item() * inputs.size(0)

    # Calculate average loss
    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss

def train_autoencoder(model, train_dataloader, testing_dataloader=None, num_epochs=10, learning_rate=1e-3, device=None):
    """
    Train an autoencoder using Mean Squared Error (MSE) loss with a progress bar for epochs.

    Parameters:
    - model: PyTorch model (autoencoder) to train.
    - train_dataloader: PyTorch DataLoader for training data.
    - testing_dataloader: PyTorch DataLoader for testing data. Default is None.
    - num_epochs: Number of training epochs. Default is 10.
    - learning_rate: Learning rate for the optimizer. Default is 1e-3.
    - device: Device to use for training ('cuda' or 'cpu'). Defaults to CUDA if available.

    Returns:
    - model: Trained autoencoder model.
    - training_losses: List of training losses for each epoch.
    - testing_losses: List of testing losses for each epoch (if testing_dataloader is provided).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_losses = []
    testing_losses = []

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # Train for one epoch
        train_loss = run_model(model, train_dataloader, criterion, device, optimizer)
        training_losses.append(train_loss)

        # Evaluate on testing data if provided
        if testing_dataloader is not None:
            test_loss = run_model(model, testing_dataloader, criterion, device)
            testing_losses.append(test_loss)

        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}" +
                   (f", Test Loss: {test_loss:.6f}" if test_loss is not None else ""))
            

    return model, training_losses, testing_losses if testing_dataloader else None



def save_results(model, training_losses, testing_losses, save_path):
    """
    Save the model and training/testing losses to a file.

    Parameters:
    - model: Trained PyTorch model.
    - training_losses: List of training losses.
    - testing_losses: List of testing losses.
    - save_path: Path to save the model and losses.
    """
    torch.save(model.state_dict(), save_path + 'model.pth')
    with open(save_path + 'training_losses.txt', 'w') as f:
        for loss in training_losses:
            f.write(f"{loss}\n")
    if testing_losses:
        with open(save_path + 'testing_losses.txt', 'w') as f:
            for loss in testing_losses:
                f.write(f"{loss}\n")