import h5py
import json
import numpy as np
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
    train_test_split=configs['data']['train_test_split']
)

# dl.create_clean_datafile()

with h5py.File(configs['data']['filename_clean'], 'r') as hf:
    nrows = hf['x'].shape[0]
    ncols = hf['x'].shape[2]

    print(hf['x'].shape)
    print(hf['y'].shape)

ntrain = int(configs['data']['train_test_split'] * nrows)
steps_per_epoch = int(ntrain / configs['data']['batch_size'])

# ntrain를 batch_size의 배수로 만들어 경계 값을 명확히 한다.
ntrain = steps_per_epoch * configs['data']['batch_size']
print('> Clean data has', nrows, 'data rows. Training on', ntrain, 'rows with', steps_per_epoch, 'steps-per-epoch')

# Building a model
sess =tf.Session()
model = lstm.LSTM(sess, configs['data']['x_window_size'], ncols,
                     configs['model']['dirname_save_model'])

sess.run(tf.global_variables_initializer())

# Train the model
# data_gen_train = dl.generate_clean_data(0, ntrain)
# model.training(configs['model']['epochs'], steps_per_epoch, data_gen_train, save=True)

# Load a trained model
model.load_model('epoch0_loss6.74e-01')

ntest = nrows - ntrain
steps_test = int(ntest / configs['data']['batch_size'])
print('> Testing model on', ntest, 'data rows with', steps_test, 'steps')

# Test the model
data_gen_test = dl.generate_clean_data(-ntest, -1)
true_values = []
predictions = model.predict(steps_test, data_gen_test, true_values)

# Print stats
s1 = stats.STATS(true_values[-ntest:], predictions[-ntest:], threshold=0.5)
s1.print_stats()
s1 = stats.STATS(true_values[-ntest:], predictions[-ntest:], threshold=0.6)
s1.print_stats()
s1 = stats.STATS(true_values[-ntest:], predictions[-ntest:], threshold=0.7)
s1.print_stats()
s1 = stats.STATS(true_values[-ntest:], predictions[-ntest:], threshold=0.75)
s1.print_stats()
# s1 = stats.STATS(true_values[-ntrain:], predictions[-ntrain:], threshold=0.8)
# s1.print_stats()

# plot the result
plot.plot_results(true_values[-ntest:], predictions[-ntest:])
