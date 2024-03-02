import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(2, 128*128)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 128)  # Reshape to match the desired output size
        return x
    

def load_data(batch_size=8):
    data_x, data_y = np.load('data_x.npy'), np.load('data_y.npy')

    inputs = torch.tensor(data_x, dtype=torch.float32)
    outputs = torch.tensor(data_y, dtype=torch.float32)
    # print(inputs.size())
    # print(outputs.size())
    # Split data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Create DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    return train_loader, test_loader


def train_model(model, train_loader, criterion = nn.MSELoss(),num_epochs=100):
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate the loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
        
        # Print average loss for the epoch
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

    print("Training finished")
    return model


def eval_model(model, test_loader,criterion = nn.MSELoss()):
    # Evaluation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient tracking during evaluation
        for X_test, y_test in test_loader:
            # Forward pass
            y_pred = model(X_test)
            
            # Calculate the loss
            val_loss += criterion(y_pred, y_test).item()

    # Calculate average validation loss
    average_val_loss = val_loss / len(X_test)
    print(f"Validation Loss: {average_val_loss}")
    return average_val_loss


def main():
    model = LinearModel()
    train_loader, test_loader = load_data()
    model = train_model(model, train_loader, test_loader, num_epochs=100)
    eval_model(model, test_loader)
    return


if __name__ == '__main__':
    main()