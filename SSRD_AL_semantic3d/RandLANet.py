
from os import makedirs
import time
from os import makedirs

import tensorflow as tf
from sklearn.metrics import confusion_matrix

import helper_tf_util

from semantic3d_dataset_train import *
from semantic3d_dataset_test3 import *
from helper_ply import write_ply

def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)

early_stop_count = 6

class  Network:
    def __init__(self, config, dataset_name, sampler_args, test_area_idx, reg_strength):
        self.config = config
        self.dataset_name = dataset_name
        self.sampler_args = sampler_args
        self.test_area_idx = test_area_idx
        self.reg_strength = reg_strength

        self.Log_file = open(join("record_log", 'log_train_' + dataset_name + "_" + str(test_area_idx) + "_" + get_sampler_args_str(sampler_args) +"_"+str(reg_strength)+ '.txt'), 'a')
        self.init_input()
        self.training_epoch = 0
        self.correct_prediction = 0
        self.accuracy = 0
        self.class_weights = DP.get_class_weights(dataset_name)

        self.training_step = 1

        with tf.variable_scope('layers', reuse=tf.AUTO_REUSE):
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.logits_3d, self.last_second_features = self.inference(self.is_training)

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            self.logits = tf.reshape(self.logits_3d, [-1, config.num_classes])
            self.last_second_features = tf.reshape(self.last_second_features, [-1, 32])

            self.labels = tf.reshape(self.input_labels, [-1])
            self.activation = tf.reshape(self.input_activation, [-1])
            self.pseudo = tf.reshape(self.input_pseudo, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx_init = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_idx = tf.reshape(valid_idx_init, [-1])

            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_activation = tf.gather(self.activation, valid_idx, axis=0)

            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)
            valid_pseudo_init = tf.gather(self.pseudo, valid_idx, axis=0)
            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)
            valid_pseudo = tf.gather(reducing_list, valid_pseudo_init)


            self.loss = self.get_loss(valid_logits, valid_pseudo, valid_activation, self.class_weights)

        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results', reuse=tf.AUTO_REUSE):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        self.saving_path = join("./data", dataset_name, str(reg_strength), "saver", get_sampler_args_str(self.sampler_args), "snapshots")
        makedirs(self.saving_path) if not exists(self.saving_path) else None
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()

        self.tensorboard_path = join("./data", dataset_name, str(reg_strength), "saver", get_sampler_args_str(self.sampler_args), "tensorboard")
        makedirs(self.tensorboard_path) if not exists(self.tensorboard_path) else None

        self.train_writer = tf.summary.FileWriter(self.tensorboard_path, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def restore_model(self, round_num):
        if round_num == 1:

            restore_snap = join("./data", self.dataset_name, str(self.reg_strength), "saver", "seed", "snapshots", 'snap-{:d}'.format(1))
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from seed")

        # Load trained model
        else:
            restore_snap = join(self.saving_path, 'snap-{:d}'.format(round_num))
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

    def restore_baseline_model(self):

        restore_snap = join("./data", self.dataset_name, str(self.reg_strength), "saver", "baseline", "snapshots", 'snap-{:d}'.format(1))
        self.saver.restore(self.sess, restore_snap)
        print("Model restored from baseline")

    def init_input(self):
        with tf.variable_scope('inputs', reuse=tf.AUTO_REUSE):
            self.input_xyz, self.input_neigh_idx, self.input_sub_idx, self.input_interp_idx = [], [], [], []
            for i in range(self.config.num_layers):
                self.input_xyz.append(tf.placeholder(tf.float32, shape=[None, None, 3]))  #[batch, point, 3]
                self.input_neigh_idx.append(tf.placeholder(tf.int32, shape=[None, None, self.config.k_n]))  #[batch, point, 16]
                self.input_sub_idx.append(tf.placeholder(tf.int32, shape=[None, None, self.config.k_n]))  #[batch, point, 16]
                self.input_interp_idx.append(tf.placeholder(tf.int32, shape=[None, None, 1]))  #[batch, point, 3]
            self.input_features = tf.placeholder(tf.float32, shape=[None, None, 6])  #[batch, point, 3+3]
            self.input_labels = tf.placeholder(tf.int32, shape=[None, None])  # [batch, point]
            self.input_activation = tf.placeholder(tf.int32, shape=[None, None])  # [batch, point]
            self.input_pseudo = tf.placeholder(tf.int32, shape=[None, None])  # [batch, point]
            self.input_input_inds = tf.placeholder(tf.int32, shape=[None, None])  # [batch, point]
            self.input_cloud_inds = tf.placeholder(tf.int32, shape=[None])  # [batch]

    def inference(self, is_training):

        d_out = self.config.d_out
        feature = self.input_features  # 就是 rgb  shape=[batch_size, point_num, 3]
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)


        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_block(feature, self.input_xyz[i], self.input_neigh_idx[i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)
            f_sampled_i = self.random_sample(f_encoder_i, self.input_sub_idx[i])
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)

        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0',
                                        [1, 1], 'VALID', True, is_training)

        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(feature, self.input_interp_idx[-j - 1])
            f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),
                                                          f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                          'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                          is_training=is_training)
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)

        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])
        return f_out, f_layer_fc2

    def get_feed_dict(self, dat, is_training):
        feed_dict = {self.is_training: is_training}
        # print("----------------------")
        for j in range(self.config.num_layers):
            feed_dict[self.input_xyz[j]] = np.squeeze(dat[j].numpy(), axis=1)  # [batch, point, 3]
            feed_dict[self.input_neigh_idx[j]] = np.squeeze(dat[self.config.num_layers + j].numpy(), axis=1)  # [batch, point, 16]
            feed_dict[self.input_sub_idx[j]] = np.squeeze(dat[2 * self.config.num_layers + j].numpy(), axis=1)  # [batch, point, 16]
            feed_dict[self.input_interp_idx[j]] = np.squeeze(dat[3 * self.config.num_layers + j].numpy(), axis=1)  # [batch, point, 3]



        feed_dict[self.input_features] = np.squeeze(dat[4 * self.config.num_layers+0].numpy(), axis=1)  # [batch, point, 3+3]
        feed_dict[self.input_labels] = np.squeeze(dat[4 * self.config.num_layers+1].numpy(), axis=1)  # [batch, point]
        feed_dict[self.input_activation] = np.squeeze(dat[4 * self.config.num_layers+2].numpy(), axis=1)  # [batch, point]
        feed_dict[self.input_pseudo] = np.squeeze(dat[4 * self.config.num_layers+3].numpy(), axis=1)  # [batch, point]
        feed_dict[self.input_input_inds] = np.squeeze(dat[4 * self.config.num_layers+4].numpy(), axis=1)  # [batch, point]
        feed_dict[self.input_cloud_inds] = np.squeeze(dat[4 * self.config.num_layers+5].numpy(), axis=1)  # [batch]



        return feed_dict

    def get_feed_dict_sub(self, dat_sub):
        feed_dict = {self.is_training: False}
        # print("----------------------")
        for j in range(self.config.num_layers):
            feed_dict[self.input_xyz[j]] = np.squeeze(dat_sub[j].numpy(), axis=1)  # [batch, point, 3]
            feed_dict[self.input_neigh_idx[j]] = np.squeeze(dat_sub[self.config.num_layers + j].numpy(), axis=1)  # [batch, point, 16]
            feed_dict[self.input_sub_idx[j]] = np.squeeze(dat_sub[2 * self.config.num_layers + j].numpy(), axis=1)  # [batch, point, 16]
            feed_dict[self.input_interp_idx[j]] = np.squeeze(dat_sub[3 * self.config.num_layers + j].numpy(), axis=1)  # [batch, point, 3]

        feed_dict[self.input_features] = np.squeeze(dat_sub[4 * self.config.num_layers+0].numpy(), axis=1)  # [batch, point, 3+3]
        feed_dict[self.input_labels] = np.squeeze(dat_sub[4 * self.config.num_layers+1].numpy(), axis=1)  # [batch, point]
        feed_dict[self.input_activation] = np.squeeze(dat_sub[4 * self.config.num_layers+2].numpy(), axis=1)  # [batch, point]
        feed_dict[self.input_pseudo] = np.squeeze(dat_sub[4 * self.config.num_layers+3].numpy(), axis=1)  # [batch, point]
        feed_dict[self.input_input_inds] = np.squeeze(dat_sub[4 * self.config.num_layers+4].numpy(), axis=1)  # [batch, point]
        feed_dict[self.input_cloud_inds] = np.squeeze(dat_sub[4 * self.config.num_layers+5].numpy(), axis=1)  # [batch]

        return feed_dict



    def get_feed_dict_train(self, dat):
        feed_dict = {self.is_training: True}
        for j in range(self.config.num_layers):
            feed_dict[self.input_xyz[j]] = dat[j]  # [batch, point, 3]
            feed_dict[self.input_neigh_idx[j]] = dat[self.config.num_layers + j]  # [batch, point, 16]
            feed_dict[self.input_sub_idx[j]] = dat[2 * self.config.num_layers + j]  # [batch, point, 16]
            feed_dict[self.input_interp_idx[j]] = dat[3 * self.config.num_layers + j]  # [batch, point, 3]
        feed_dict[self.input_features] = dat[4 * self.config.num_layers + 0]  # [batch, point, 3+3]
        feed_dict[self.input_labels] = dat[4 * self.config.num_layers + 1]  # [batch, point]
        feed_dict[self.input_activation] = dat[4 * self.config.num_layers + 2]  # [batch, point]
        feed_dict[self.input_pseudo] = dat[4 * self.config.num_layers + 3]  # [batch, point]
        feed_dict[self.input_input_inds] = dat[4 * self.config.num_layers + 4]  # [batch, point]
        feed_dict[self.input_cloud_inds] = dat[4 * self.config.num_layers + 5]  # [batch]

        return feed_dict

    def get_feed_dict_test(self, dat):
        feed_dict = {self.is_training: False}
        for j in range(self.config.num_layers):
            feed_dict[self.input_xyz[j]] = dat[j]  # [batch, point, 3]
            feed_dict[self.input_neigh_idx[j]] = dat[self.config.num_layers + j]  # [batch, point, 16]
            feed_dict[self.input_sub_idx[j]] = dat[2 * self.config.num_layers + j]  # [batch, point, 16]
            feed_dict[self.input_interp_idx[j]] = dat[3 * self.config.num_layers + j]  # [batch, point, 3]
        feed_dict[self.input_features] = dat[4 * self.config.num_layers+0]  # [batch, point, 3+3]
        feed_dict[self.input_labels] = dat[4 * self.config.num_layers+1]  # [batch, point]
        feed_dict[self.input_input_inds] = dat[4 * self.config.num_layers+2]  # [batch, point]
        feed_dict[self.input_cloud_inds] = dat[4 * self.config.num_layers+3]  # [batch]

        return feed_dict

    def reset_lr(self):
        op = self.learning_rate.assign(self.config.learning_rate)
        self.sess.run(op)


    def train2(self, round_num):
        self.reset_lr()

        self.training_epoch = 0
        log_out("Round " + str(round_num) + ' | ****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        best_miou = 0
        best_OA = 0
        train_data, test_data, test_probs = None, None, None
        if self.dataset_name == "semantic3d":
            train_data = Semantic3D_Dataset_Train(reg_strength=self.reg_strength, sampler_args=self.sampler_args,
                                                  round_num=round_num)
            test_data = Semantic3D_Dataset_Test()
            test_probs = [np.zeros(shape=[l.shape[0], self.config.num_classes], dtype=np.float32) for l in
                          test_data.input_labels]

        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            activation_sum = 0
            dat = train_data.get_batch()
            while len(dat) > 0:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.activation,
                       self.accuracy]
                _, _, summary, l_out, probs, labels, acti, acc = self.sess.run(ops, feed_dict=self.get_feed_dict_train(dat))

                activation_sum = activation_sum + np.sum(acti)

                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

                dat = train_data.get_batch()


            log_out("Round " + str(round_num) + ' | epoch=' + str(self.training_epoch) + ", train costTime=" + str(
                time.time() - t_start) + ", | total_activation_sum=" + str(activation_sum), self.Log_file)
            self.training_epoch += 1
            # Update learning rate
            op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                       self.config.lr_decays[self.training_epoch]))
            self.sess.run(op)

            if self.training_epoch >= int(self.config.max_epoch * 0.6):
                tt12 = time.time()
                if self.dataset_name == "semantic3d":
                    m_iou, OA = self.evaluate_test_semantic3d(dataset=test_data, test_probs=test_probs)

                log_out("Round " + str(round_num) + ' | epoch=' + str(self.training_epoch) + ", current m_iou=" + str(m_iou), self.Log_file)
                if m_iou > best_miou: