import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator


class Histogram:
    def __init__(self, P_NG, P_OK, output_folder="Histogram", filename='Histogram_test.png'):
        self.P_NG = np.array(P_NG)
        self.P_OK = np.array(P_OK)
        self.output_folder = output_folder
        self.plot_histograms(self.output_folder, filename)

    def plot_histograms(self, output_folder, filename):
        os.makedirs(output_folder, exist_ok=True)
        plt.figure(figsize=(12, 6))

        # Plot histogram for P_NG
        plt.subplot(1, 2, 1)
        plt.hist(self.P_NG[:], bins=20, range=(0, 1), color='red', alpha=0.7, label='NG (label 0, 1, 2)')
        plt.xlabel('Softmax Probability (NG label)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Softmax Probabilities for NG (label 0, 1, 2)')
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()

        # Plot histogram for P_OK
        plt.subplot(1, 2, 2)
        plt.hist(self.P_OK[:], bins=20, range=(0, 1), color='green', alpha=0.7, label='OK (label 3)')
        plt.xlabel('Softmax Probability (OK label)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Softmax Probabilities for OK (label 3)')
        plt.legend()

        plt.tight_layout()
        save_path = os.path.join(output_folder, filename)
        plt.savefig(save_path)

        print(f'Saved filename: {filename} in output_folder: {output_folder}')


if __name__ == "__main__":
    # NG = [0.01, 0.13, 0.01, 0.23, 0.16, 0.0, 0.35, 0.25, 0.69, 0.12, 0.13, 0.04, 0.21, 0.33, 0.7, 0.15, 0.19, .19, .19, .37, .13, .11, .37, .12, .36, .09, .07, 0.31, .41, .08, .08, 0.83, .15, .54, .47, .19, .41]
    NG = [0.02, .35, .48, .37, .5, .12, .41, .28, .28, .69, .47, .52, .32, .34, .29, .06, 0.28, .18, .45, .222, .2, .65, .36, .09, .35, .17, .33, .57, .28, .19, .24, .37, .13, .48, .34, .55]
    OK = [1]
    Histogram(NG, OK, output_folder="Histogram", filename='Hist')