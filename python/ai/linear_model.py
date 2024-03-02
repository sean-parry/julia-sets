import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import get_data

class LinearModel(nn.Module):
    def __init__(self, width, height):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(2, width*height)
        self.width = width
        self.height = height

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.width, self.height)  # Reshape to match the desired output size
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

def generate_data(amount,width=128,height=128, batch_size=8):
    data_x, data_y = get_data.get(amount,width,height)

    inputs = torch.tensor(data_x, dtype=torch.float32)
    outputs = torch.tensor(data_y, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Create DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    # can change below shuffle to true to verify that dif batches are being used in training (they are don't worry)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False) 
    return train_loader, test_loader


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

def main():
    # this linear network is only one layer it gets the basic shape but i think the loss function isn't very effective
    # and improve the model architecture ie use cnns
    # i should add one of those pytorch progress bars that people use as well
    # anyway all of this code is fine i should just make sure i am saving the models with a meaningful name
    # make the model a bit better for linear, (add layers and try some loss functions out)
    # make a cnn model
    # make sure both are tuned reasonably
    # save a 360p by 360p version of both models
    # deploy them onto a website with linear, cnn, real, input ( for an input (mouse click on the argand plain))
    # then do a quick write up about the different methods

 
    batch_size = 8
    num_epochs = 100
    width, height = 100,100

    model = LinearModel(width, height)

    # this way of using train_loader and test_loader is completely fine
    train_loader, test_loader = generate_data(155,width=width,height=height, batch_size=batch_size)
    print(train_loader) # this prints out that its an object if u wanted confimation
    model = train_model(model, train_loader, num_epochs=num_epochs)
    eval_model(model, test_loader)
    
    X_test, y_test, y_pred = get_preditions(model, test_loader)
    
    for i in range(len(X_test)):
        plot_predictions(y_pred[i], y_test[i])



if __name__ == '__main__':
    main()