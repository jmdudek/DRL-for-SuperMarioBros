import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.style.use('ggplot') 


def str_to_bool(string: str):
    """Function to convert string to boolean
    
    Arguments:
        string (str): String to convert
        
    Returns
        (bool): The boolean that is returned
    """
    
    if string == 'True':
         return True
    elif string == 'False':
         return False


def evaluation(data_paths: list, labels: list, mode: str, avg_window: int=500, save: bool=False, save_path: str=""):
    """Function to plot the data from our transfer learning experiments
    
    Arguments:
        data_paths (list): Paths to load the plotting data from
        labels (list): Labels corresponding to the datasets that should be loaded
        mode (str): Determines the plotting mode, needs to be one of the following: ["loss", "reward", "win_rate"]
        avg_window (int): Window size for the averaging 
        save (bool):  Whether to save the resulting plot or not
        save_path (str): Path for saving the evaluation plot
    """
    
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize = (10, 6))
    
    # Create the plot given the mode
    if mode == "loss":
        ax1.set(ylabel='Losses', xlabel='Epochs', title=f'Average Loss')
    elif mode  == "reward":
        ax1.set(ylabel='Cumul Reward', xlabel='Episodes', title=f'Cumulative Reward')
    else:
        ax1.set(ylabel='Win Rate', xlabel='Episodes', title=f'Average Win Rate')
    
    # Loop over all datasets and add each to the plot
    for idx, path in enumerate(data_paths):
        
        # Read data
        with open(f"{path}", encoding='utf8', newline="") as input_file:
            reader = csv.reader(input_file)
            plotting_data = list(reader)
    
            # Loss plot
            if mode == "loss":
                data = [float(entry) for entry in plotting_data[0]]
                ax1.plot(np.convolve(np.asarray(data), np.ones(avg_window)/avg_window, mode='valid'), label=labels[idx])
            # Reward plot (will output a moving average)
            elif mode  == "reward":
                data = [float(entry) for entry in plotting_data[1]]
                ax1.plot(np.convolve(np.asarray(data)*600, np.ones(avg_window)/avg_window, mode='valid'), label=labels[idx])
            # Win plot (will output a moving average)
            else:
                data = [str_to_bool(entry) for entry in plotting_data[2]]
                ax1.plot(np.convolve(data, np.ones(avg_window)/avg_window, mode='valid'), label=labels[idx])
    
    ax1.legend()
    
    if save:
        plt.savefig(f"{save_path}", dpi=500.0, format="png")
        
    plt.show()