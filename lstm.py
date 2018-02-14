# coding=utf-8
import numpy as np
import tensorflow as tf
import datetime
import os

class LSTM:

    def __init__(self, session: tf.Session, x_window_size: int, ncols: int,
               dirname_save_model: str, class_weight=0.5, name: str="main") -> None:
        """
        Args:
          session (tf.Session): Tensorflow session
          x_window_size: sequence length
          ncols: number of columns (e.g. open,high,low,close -> 4)
          name (str, optional): TF Graph will be built under this name scope
        """

        self.session = session
        self.x_window_size = x_window_size
        self.ncols = ncols
        self.net_name = name
        self.save_path = dirname_save_model

        self.class_weight = class_weight

        self._build_network()

    def _build_network(self, l_rate=0.001) -> None:
        """
        # Input -> LSTM layer -> FC layer -> Output
        Args:
          l_rate (float, optional): Learning rate
        """
        with tf.variable_scope(self.net_name):

            # [batch_size(None), window_size, num of cols]
            self.X = tf.placeholder(tf.float32, [None, self.x_window_size, self.ncols], name='x')

            # [batch_size(batch_size(None), output_dim]
            self.Y = tf.placeholder(tf.float32, [None, 1], name='y')

            # dropout keep prob
            self.output_keep_prob = tf.placeholder(tf.float32, name='output_keep_prob')

            cell = []
            cell_1 = tf.contrib.rnn.LSTMCell(num_units=100, state_is_tuple=True, use_peepholes=True)
            cell_1 = tf.contrib.rnn.DropoutWrapper(cell_1, output_keep_prob=self.output_keep_prob)
            cell.append(cell_1)

            cell = tf.contrib.rnn.MultiRNNCell(cell)

            outputs, states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32, scope='rnn1')

            # # outputs[:, -1] 는 LSTM layer의 output 중에서 마지막 것만 사용한다는 의미.
            self.Y_pred = tf.contrib.layers.fully_connected(
                outputs[:, -1], 1, activation_fn=tf.nn.sigmoid, scope='fc1')

            # cost/loss
            self.loss = -tf.reduce_mean(self.Y * self.class_weight * tf.log(tf.clip_by_value(self.Y_pred, 0.1, 1.0))
                                        + (1 - self.Y) * tf.log(tf.clip_by_value(1 - self.Y_pred, 0.1, 1.0)))

            # optimizer
            self.optimizer = tf.train.AdamOptimizer(l_rate)
            self.train = self.optimizer.minimize(self.loss)

            # RMSE
            self.targets = tf.placeholder(tf.float32, [None, 1])
            self.predictions = tf.placeholder(tf.float32, [None, 1])
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))

    def training(self, epochs, steps_per_epoch, data_gen_train, save=False):
        self.session.run(tf.global_variables_initializer())

        # for debugging
        all_vars = tf.global_variables()
        def get_var(name):
            for i in range(len(all_vars)):
                if all_vars[i].name.startswith(name):
                    return all_vars[i]
            return None
        fc1_var = get_var('main/fc1/weights')
        rnn1_var = get_var('main/rnn1')

        step_loss=0
        for i in range(epochs):
            for j in range(steps_per_epoch):
                batch_x, batch_y = next(data_gen_train)
                batch_y = np.array(batch_y).reshape(-1, 1)

                _, step_loss = self.session.run([self.train, self.loss], feed_dict={
                    self.X: batch_x, self.Y: batch_y, self.output_keep_prob: 0.8})
                train_predict, fc1_var_np, rnn1_var_np = self.session.run([self.Y_pred, fc1_var, rnn1_var],
                                                feed_dict={self.X: batch_x, self.output_keep_prob: 1.0})
                # print(train_predict)
                print("[epoch: {}, step: {}] loss: {}".format(i, j, step_loss))

            if save:
                # save model at every epoch
                self.save_model('epoch{}_loss{:.2e}'.format(i, step_loss), write_meta_graph=(i==0))

    def predict(self, steps_test, data_gen_test, true_values):
        rmse_list = np.array([])
        test_predict_list = np.array([])

        for i in range(steps_test):
            test_batch_x, test_batch_y = next(data_gen_test)
            # save true_value(targets) for plotting a graph in main
            true_values += list(test_batch_y)
            test_batch_y = np.array(test_batch_y).reshape(-1, 1)
            test_predict = self.session.run(self.Y_pred,
                                            feed_dict={self.X: test_batch_x, self.output_keep_prob: 1.0})
            rmse_val = self.session.run(self.rmse, feed_dict={
                self.targets: test_batch_y, self.predictions: test_predict})

            rmse_list = np.append(rmse_list,rmse_val)
            test_predict_list = np.append(test_predict_list, test_predict)

        print("> RMSE : {}".format(rmse_list.mean()))
        return test_predict_list

    def save_model(self, modelname='', write_meta_graph=False):
        saver = tf.train.Saver()
        if modelname == '':
            # make today's file name
            today = datetime.date.today()
            today = today.strftime('%y%m%d')
            modelname = 'model' + today

        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path_full = os.path.join(save_path, modelname)
        saver.save(self.session, save_path_full, write_meta_graph=write_meta_graph)

    def load_model(self, modelname):
        load_path = self.save_path
        meta_filename = modelname + '.meta'
        ckpt_filename = modelname

        # meta path formatting : saved_model/modelname.meta
        meta_path_full = os.path.join(load_path, meta_filename)
        # ckpt path formatting : saved_model/modelname (error occur if + '.ckpt')
        ckpt_path_full = os.path.join(load_path, ckpt_filename)

        print('> Load checkpoint : {}'.format(ckpt_path_full))
        saver = tf.train.Saver()

        # tf.reset_default_graph()
        # saver = tf.train.import_meta_graph(meta_path_full)

        saver.restore(self.session, ckpt_path_full)
