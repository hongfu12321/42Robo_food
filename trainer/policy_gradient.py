import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf

import datetime 
# reproducible
np.random.seed(42)
tf.set_random_seed(42)

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=True,
            saving_model=True,
            model_name='default',
            output_dir='/tmp/pizza_ai',
            job_dir='/tmp/pizza_ai',
    ):
        # self.epoch_counter = 0
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.saving_model = saving_model
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        #self.metrics = Metrics()

        self._build_net()

        self.sess = tf.Session()

        if saving_model:
            self.model_name = model_name
            self.saver = tf.train.Saver()
            self.model_path = job_dir
            #self.saver.restore(self.sess, "./models/default5.ckpt")

        if output_graph:
            # IAN ADD, self.writer
            self.job_dir = job_dir
            self.writer = tf.summary.FileWriter(job_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer_h1 = tf.layers.dense(
            inputs=self.tf_obs,
            units=42,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )

        layer_h2 = tf.layers.dense(
            inputs=layer_h1,
            units=21,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        # fc2
        all_act = tf.layers.dense(
            inputs=layer_h2,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc3'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
            # IAN ADD
            self.loss_sum = tf.summary.scalar(name='loss summary per epoch', tensor=loss)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    #def add_metrics(self, gameID):
    #    self.metrics.statistical(self.ep_as, self.ep_rs, self.ep_obs, gameID)
    #    self.metrics.most_common_actions(self.ep_as, gameID)

    def clear_rollout(self):
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def save_model(self, epc):
        print('Saving model to: ', self.model_path)
        # self.saver.save(self.sess, self.save_name + str(self.epoch_counter) + '.ckpt')
        self.saver.save(self.sess, self.model_path + '/' + self.model_name + '-' + datetime.datetime.now().isoformat() + '.ckpt')
        # self.epoch_counter+=1

    def restore_model(self, model_name):
        restore_path = self.model_path + '/' + self.model_name + '.ckpt'
        self.saver.restore(self.sess, restore_path)
        print("Restoring model from: ", restore_path)

    def learn(self, epc):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode

        # IAN ADD, loss_sum
        train_op, loss_sum = self.sess.run([self.train_op, self.loss_sum], feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })
        # IAN ADD
        self.writer.add_summary(loss_sum, epc)
        if self.saving_model:
           self.save_model(epc)
        #self.clear_rollout()   # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs = np.subtract(discounted_ep_rs, np.mean(discounted_ep_rs))
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
