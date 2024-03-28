import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def train_model(model, train_loader, num_epochs=100, criterion = nn.MSELoss()):
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

def get_preditions(model, test_loader, return_as_tensor = False):
    model.eval()
    res = []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            # Forward pass
            y_pred = model(X_test)
            clipped_y_pred = torch.clamp(y_pred,0,1)
            res.append([X_test,y_test,clipped_y_pred])
    if return_as_tensor:
        return res
    return res[0][0].numpy(), res[0][1].numpy(), res[0][2].numpy()

def plot_predictions(output,actual):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 

    # Plot predicted output
    axes[0].imshow(output)
    axes[0].axis('off')
    axes[0].set_title('Predicted Output')

    # Plot actual values
    axes[1].imshow(actual)
    axes[1].axis('off')
    axes[1].set_title('Actual Values')

    plt.show()

def save_model(model):
    torch.save(model,'model.pkl')