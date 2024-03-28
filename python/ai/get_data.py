import numpy as np , matplotlib.pyplot as plt, pandas as pd, random, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

'''
at this point why not just write an ai that tries to predict the output for each julia set because why not
apply ai to stuff that doesn't need ai applied to it

I should do this for a circle on the argand diagram and make a gif that fills in the last 25% and then generate a mp3 / git or something
'''
def plot_data(df):
    num_plots = len(df)
    rows = math.isqrt(num_plots)
    cols = math.ceil(num_plots / rows)

    fig, axs = plt.subplots(rows, cols, figsize=(cols*6, rows*6))

    if num_plots == 1:
        axs = [[axs]]

    for i in range(rows):
        for j in range(cols):
            if i * cols + j >= num_plots:  # In case we have more subplots than needed
                break
            axs[i][j].imshow(df['output'][i * cols + j]) # if you want to change theme cmap = here
            axs[i][j].set_title(f'Input: {df["input"][i * cols + j]}')
            axs[i][j].axis('off')

    plt.tight_layout()
    plt.show()

def _generate_data(input: tuple[float],
                   width: int,
                   height: int,
                   max_iter:int
                   ) -> list[float]:
    
    re, im = input
    c = re+im*1j
    y, x = np.ogrid[1.4: -1.4: height*1j, -1.4: 1.4: width*1j]
    z_array = x + y*1j
    iter_until_diverge = max_iter + np.zeros(z_array.shape)
    # create array of all True
    not_already_diverged = iter_until_diverge < 10000

	# creat array of all False
    diverged_in_past = iter_until_diverge > 10000

    for i in range(max_iter):
        z_array = z_array**2 + c
        z_size_array = z_array * np.conj(z_array)
        diverging = z_size_array > 4
		
        diverging_now = diverging & not_already_diverged
        iter_until_diverge[diverging_now] = i
		
        not_already_diverged = np.invert(diverging_now) & not_already_diverged
        diverged_in_past = diverged_in_past | diverging_now
        z_array[diverged_in_past] = 0

    return(iter_until_diverge/100)

def _get(amount: int,
         width: int = 128,
         height: int = 128,
         max_iter: int = 100,
         verbose: bool = True
         ) -> list[list[float]]:

    input, output = [], []
    for _ in range (amount):
        input.append([round(random.uniform(-0.999,.999),3) for _ in range (2)])
        if verbose:
            print(f'calculating julia set for {input[-1][0]} + {input[-1][1]}j')
        output.append(_generate_data(input[-1],width,height,max_iter))

    return input, output


def save_data(amount: int,
              width: int = 128,
              height: int = 128, 
              max_iter: int = 100
              ) -> None:
    
    input, output = _get(amount, width, height, max_iter)
    np.save('data_x.npy',input)
    np.save('data_y.npy',output)

def load_data(batch_size: int=8
              ) -> tuple[DataLoader, DataLoader]:
    
    data_x, data_y = np.load('data_x.npy'), np.load('data_y.npy')

    inputs = torch.tensor(data_x, dtype=torch.float32)
    outputs = torch.tensor(data_y, dtype=torch.float32)

    # Split data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Create DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    return train_loader, test_loader

def get_data_loaders(amount: int,
                     width: int = 128,
                     height: int = 128,
                     batch_size: int = 8
                     ) -> tuple[DataLoader, DataLoader]:
    
    data_x, data_y = _get(amount,width,height)

    inputs = torch.tensor(data_x, dtype=torch.float32)
    outputs = torch.tensor(data_y, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    # Create DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    # can change below shuffle to true to verify that dif batches are being used in training (they are don't worry)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False) 
    return train_loader, test_loader

def main():
    get_data_loaders(10)
    return

if __name__ == '__main__':
    main()