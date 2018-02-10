import matplotlib.pyplot as plt
import matplotlib.style as style
import json
import pandas as pd

style.use('seaborn')

def plot_results(true_data, predicted_data):
    fig = plt.figure(0)
    # ax = fig.add_subplot(111)

    for i in range(len(true_data)):
        # ax.plot(i,true_data[i], 'o', label='True Data', ms=3, color="black")

        if true_data[i] == 1:
            if predicted_data[i] >= 0.5:
                plt.plot(i,predicted_data[i], 'o', label='True Positive', ms=3, color='blue')
            else:
                plt.plot(i, predicted_data[i], 'o', label='False Negative', ms=3, color='red')
        else:
            if predicted_data[i] < 0.5:
                plt.plot(i,predicted_data[i], 'o', label='True Negative', ms=3, color='green')
            else:
                plt.plot(i, predicted_data[i], 'o', label='False Positive', ms=3, color='orange')

    plt.legend()
    plt.show()
