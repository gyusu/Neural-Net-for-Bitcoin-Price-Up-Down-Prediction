import h5py
import json
import pandas as pd
import tensorflow as tf

import etl, lstm, plot, stats

tf.set_random_seed(777) # for reproducibility
configs = json.loads(open('configs.json').read())

dl = etl.ETL(
    filename_in=configs['data']['filename'],
    filename_out=configs['data']['filename_clean'],
    batch_size=configs['data']['batch_size'],
    x_window_size=configs['data']['x_window_size'],
    y_window_size=configs['data']['y_window_size'],
    x_col=configs['data']['x_base_column'],
    y_col=configs['data']['y_predict_column'],
    filter_cols=configs['data']['filter_columns'],
    target_percent_change=configs['rule']['target_percent_change'],
    losscut_percent_change=configs['rule']['losscut_percent_change'],
    train_test_split=configs['data']['train_test_split']
)

ans_create_data = input("DO YOU WANT TO CREATE NEW DATAFILE ? (y / n)")
if configs['model']['filename_load_model']:
    ans_load_model = input('DO YOU WANT TO LOAD TRAINED MODEL ? \'{}\' (y / n)'
                           .format(configs['model']['filename_load_model']))
else:
    ans_load_model = False

if ans_create_data in ['y','Y']:
    dl.create_clean_datafile()

with h5py.File(configs['data']['filename_clean'], 'r') as hf:
    nrows = hf['x'].shape[0]
    ncols = hf['x'].shape[2]
    ntrain = int(hf['ntrain'][0])

    # check the num of each class
    num_minority = sum(hf['y'][:ntrain])
    num_majority = ntrain - num_minority

    # calculate the imbalance ratio. used for construct the loss(cost) fn.
    imbalance_ratio = num_majority / num_minority

steps_per_epoch = int(ntrain / configs['data']['batch_size'])

print('> Clean data has', nrows, 'data rows. Training on', ntrain, 'rows with', steps_per_epoch, 'steps-per-epoch')
print('> Class 0: {}, Class 1: {} --in training set'.format(num_majority, num_minority))
print('> imbalance_ratio(class_weight): {}'.format(imbalance_ratio))

# Building a model
sess =tf.Session()
model = lstm.LSTM(sess, configs['data']['x_window_size'], ncols,
                     configs['model']['dirname_save_model'], class_weight=imbalance_ratio)

sess.run(tf.global_variables_initializer())

if ans_load_model in ['y', 'Y']:
    # Load a trained model
    model.load_model(configs['model']['filename_load_model'])
else:
    # Train the model
    data_gen_train = dl.generate_clean_data(0, ntrain)
    model.training(configs['model']['epochs'], steps_per_epoch, data_gen_train, save=True)

ntest = nrows - ntrain
steps_test = int(ntest / configs['data']['batch_size'])
print('> Testing model on', ntest, 'data rows with', steps_test, 'steps')

# Test the model
data_gen_test = dl.generate_clean_data(-ntest, -1)
true_values = []
predictions = model.predict(steps_test, data_gen_test, true_values)

# Print stats
s1 = stats.STATS(true_values[-ntrain:], predictions[-ntrain:], threshold=0.5)
s1.print_stats()
s1 = stats.STATS(true_values[-ntrain:], predictions[-ntrain:], threshold=0.6)
s1.print_stats()
s1 = stats.STATS(true_values[-ntrain:], predictions[-ntrain:], threshold=0.7)
s1.print_stats()
s1 = stats.STATS(true_values[-ntrain:], predictions[-ntrain:], threshold=0.8)
s1.print_stats()
s1 = stats.STATS(true_values[-ntrain:], predictions[-ntrain:], threshold=0.9)
s1.print_stats()

# Plot 1. just plot the results
plot.plot_results(true_values[-ntest:], predictions[-ntest:])

# read price data
data = pd.read_csv(configs['data']['filename'], index_col=0)
true_price_data = data['close'].values[-(ntest + configs['data']['y_window_size']):]

# Plot 2. plot the results with price data(close price)
plot.plot_results_with_price(true_values[-ntest:], predictions[-ntest:], true_price_data, threshold=0.8)
