import numpy as np , matplotlib.pyplot as plt, pandas as pd, random, math

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

def generate_data(input,width,height,max_iter):
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

def get(amount, width=128, height=128, max_iter=100):
    input, output = [], []
    for _ in range (amount):
        input.append([round(random.uniform(-0.999,.999),3) for _ in range (2)])
        print(f'calculating julia set for {input[-1][0]} + {input[-1][1]}j')
        output.append(generate_data(input[-1],width,height,max_iter))

    return input, output


def save_data(amount, width=128, height=128, max_iter=100):
    input, output = get(amount, width, height, max_iter)
    np.save('data_x.npy',input)
    np.save('data_y.npy',output)



def main():
    get(64)
    return
    df_from_pkl = pd.read_pickle('data.pkl')
    plot_data(df_from_pkl)

if __name__ == '__main__':
    main()