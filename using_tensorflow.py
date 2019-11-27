'''Trains an agent with Policy Gradients on Pong, using tensorflow/keras'''
import numpy as np
import pickle
import gym

import tensorflow as tf

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update ?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = False
D = 80 * 80 # input dimensionality: 80x80 grid


class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.relu):
        W = tf.random_normal(shape=(M1, M2)) * np.sqrt(2 / M1)
        self.W = tf.Variable(W)
        self.f = f

    def forward(self, X):
        a = tf.matmul(X, self.W)
        return self.f(a)

class PolicyModel:
    def __init__(self, D, K, hidden_layer_sizes=[]):
        '''Creates Model with input size D and output size K'''

        # create hidden layers
        M1 = D
        self.layers = []
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # final layer
        layer = HiddenLayer(M1, K, tf.nn.softmax)
        self.layers.append(layer)

        # inputs and targets
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(dtype=tf.int32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(dtype=tf.float32, shape=(None,), name='advantages')

        # computational graph for predict operation
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        p_a_given_s = Z
        self.predict_op = p_a_given_s

        # get the probabilities of the actions selected
        # e.g.: p_a_given_s = [0.25, 0.4, 0.35], actinos = [1, 1, 0, 0, 2, 1]
        # p_a_given_s * tf.one_hot(actions, 3) =
        # [[0, 0.40, 0],
        #  [0, 0.40, 0],
        #  [0.25, 0, 0],
        #  [0.25, 0, 0],
        #  [0, 0, 0.35],
        #  [0, 0.40, 0]]
        # using tf.reduce_sum gives [0.4, 0.4, 0.25, 0.25, 0.35, 0.4]
        selected_probs = tf.log(
            tf.reduce_sum(
                p_a_given_s*tf.one_hot(self.actions, K),
                reduction_indices=[1]
            )
        )
        self.selected_probs = selected_probs
        # Policy gradient cost function formula
        cost = -tf.reduce_sum(self.advantages * selected_probs)
        self.cost = cost
        self.train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

        self.saver = tf.train.Saver()

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, actions, advantages):
        # size check
        X = np.atleast_2d(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)

        # print(self.predict_op.eval(feed_dict={self.X: X, self.actions: actions, self.advantages:advantages}))

        self.session.run(
            self.train_op,
            feed_dict={self.X: X, self.actions: actions, self.advantages:advantages}
        )

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X:X})

    def sample_action(self, X):
        p = self.predict(X)[0]
        return np.random.choice(len(p), p=p)

    def save(self):
        save_path = self.saver.save(self.session, 'model/model.ckpt')

    def load(self):
        self.saver.restore(self.session, 'model/model.ckpt')


def prepo(I):
    ''' prepo 210x160x3 uint8 frame into 6400 (80x80) 1D float vector '''
    I = I[35:195] # crop
    I = I[::2, ::2, 0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel() # flatten

def discount_rewards(r):
    ''' take 1D float array of rewards and compute discounted reward'''
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def play_one_mc(env, model, render=False):
    observation = env.reset()
    prev_x = None
    done = False
    total_reward = 0

    states = []
    actions = []
    rewards = []

    reward = 0

    while not done:
        if render:
            env.render()

        # preprocess the observation, set input to network to be difference image
        cur_x = prepo(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # predict probabilities, then sample action
        action = model.sample_action(x)

        # save (s, a, r)
        states.append(x)
        actions.append(action)
        rewards.append(reward)

        observation, reward, done, info = env.step(action + 2)
        total_reward += reward

    # save the last (s, a, r)
    cur_x = prepo(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)

    action = model.sample_action(x)
    states.append(x)
    actions.append(action)
    rewards.append(reward)

    # calculate returns and advantages
    returns = []
    advantages = []
    G = 0
    for s, r in zip(reversed(states), reversed(rewards)):
        returns.append(G)
        advantages.append(G)
        G = r + gamma * G
    returns.reverse()
    advantages.reverse()

    # normalize advantages
    # advantages = np.array(advantages)
    # advantages -= np.mean(advantages)
    # advantages /= np.std(advantages)

    model.partial_fit(states, actions, advantages)

    return total_reward


def main():
    env = gym.make('Pong-v0')
    model = PolicyModel(D, 2, [H])
    session = tf.InteractiveSession()
    model.set_session(session)
    if resume:
        print('loading model...')
        model.load()
        print('model loaded.')
    else:
        init = tf.global_variables_initializer()
        model.session.run(init)

    episode_number = 0

    running_reward = None

    while True:
        total_reward = play_one_mc(env, model, render=False)
        episode_number += 1

        if episode_number % 100 == 0:
            print('Saving model...')
            model.save()
            print('Model saved.')

        running_reward = total_reward if running_reward is None else running_reward * 0.99 + total_reward * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (total_reward, running_reward))

if __name__ == '__main__':
    main()







