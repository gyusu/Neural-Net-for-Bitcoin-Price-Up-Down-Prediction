import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('seaborn')

def plot_results(true_data, predicted_data):
    """
    Input
    -----
    true_data : true data (0 or 1) (list or ndarray)
    predicted_data : predicted data (0 or 1) (list or ndarray)

    Result
    ------
    Plot the predictions
        True Positive : blue dot
        False Negative : red dot
        True Negative : green dot
        False Positive : orange dot

    * if prediction >= 0.5 -> positive, else -> negative
    """
    fig = plt.figure(0)

    label1 = 'True Positive'
    label2 = 'False Negative'
    label3 = 'True Negative'
    label4 = 'False Positive'

    for i in range(len(true_data)):

        if true_data[i] == 1:
            if predicted_data[i] >= 0.5:
                plt.plot(i,predicted_data[i], 'o', label=label1, ms=3, color='blue')
                label1 = ""
            else:
                plt.plot(i, predicted_data[i], 'o', label=label2, ms=3, color='red')
                label2 = ""
        else:
            if predicted_data[i] < 0.5:
                plt.plot(i,predicted_data[i], 'o', label=label3, ms=3, color='green')
                label3 = ""
            else:
                plt.plot(i, predicted_data[i], 'o', label=label4, ms=3, color='orange')
                label4 = ""

    plt.legend(bbox_to_anchor=(1.0, 0.9), loc='upper right', bbox_transform=plt.gcf().transFigure)
    plt.show()


def plot_results_with_price(true_data, predicted_data, price_data, threshold=0.5):
    """
    Input
    -----
    true_data : true data (0 or 1) (list or ndarray)
    predicted_data : predicted data (0 or 1) (list or ndarray)
    price_data : price data (close price) (list or ndarray)
    * len(price_data) must be len(true_data) + y_window_size

    Result
    -------
    1. plot the test close price data
    2-1. plot blue dot if prediction is true positive
    2-2. plot orange dot if prediction is false positive

    * calculating whether the prediction is positive or negative depends on threshold
    """

    fig = plt.figure(1)
    plt.suptitle('Postive/Negative threshold = {}'.format(threshold))

    ax = fig.add_subplot(111)
    ax.plot(price_data, linewidth=2, color='black')

    label1 = 'True Positive'
    label2 = 'False Positive'

    for i in range(len(true_data)):

        if true_data[i] == 1:
            if predicted_data[i] >= threshold:
                plt.plot(i, price_data[i], 'o', label=label1, ms=3, color='blue')
                label1 = ""
            else:
                pass
                # plt.plot(i, price_data[i], 'o', label='False Negative', ms=3, color='red')
        else:
            if predicted_data[i] < threshold:
                pass
                # plt.plot(i, price_data[i], 'o', label='True Negative', ms=3, color='green')
            else:
                plt.plot(i, price_data[i], 'o', label=label2, ms=3, color='orange')
                label2 = ""

    plt.legend(loc='upper right')
    plt.show()
