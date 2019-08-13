import random, math
import numpy as np
import keras.backend as K
from logic import Game
from keras import optimizers, layers
from keras import utils as np_utils
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling1D

class Agent():

	def __init__(self, input_dim, output_dim, loss_penalty=1000, invalid_move_penalty=4, discount_rate=1, conv_hidden_dims=[2048], dense_dim_factor=8):
		
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.conv_hidden_dims = conv_hidden_dims
		self.dense_dim_factor = dense_dim_factor
		self.dense_hidden_dims = self.get_dense_hidden_dims(self.conv_hidden_dims[-1]*3, dense_hidden_dims=[self.conv_hidden_dims[-1] for x in range(8)])
		self.invalid_move_penalty = invalid_move_penalty
		self.loss_penalty = loss_penalty
		self.discount_rate = discount_rate
		self.cross_game_exploration = 1
		self.cross_game_exploration_decay = 0.9999
		self.in_game_exploitation_decay = 0.049
		self.in_game_exploitation_decay_across_games = 0.9999

		self.__build_network()
		self.__build_train_function()


	def get_dense_hidden_dims(self, input_dim, dense_hidden_dims=[]):

		next_dim = int(input_dim/self.dense_dim_factor)

		if next_dim <= self.output_dim:
			return dense_hidden_dims

		else:
			dense_hidden_dims.append(next_dim)
			return self.get_dense_hidden_dims(next_dim, dense_hidden_dims)


	def __build_network(self):
		
		self.X = layers.Input(shape=list(self.input_dim))
		net = self.X

		for conv_dim in self.conv_hidden_dims:
			net = Conv2D(conv_dim, 2, strides=1, padding='same')(net)
			net = Activation("sigmoid")(net)

		net = Flatten()(net)

		for dense_dim in self.dense_hidden_dims:
			net = Dense(dense_dim)(net)
			net = Activation("sigmoid")(net)

		net = Dense(self.output_dim)(net)
		net = Activation("softmax")(net)

		self.model = Model(inputs=self.X, outputs=net)

	def __build_train_function(self):

		action_prob_placeholder = self.model.output
		action_onehot_placeholer = K.placeholder(shape=(None, self.output_dim),
														name='action_onehot')
		discount_reward_placeholder = K.placeholder(shape=(None,),
													name='discount_reward')

		action_prob = K.sum(action_prob_placeholder*action_onehot_placeholer, axis=1)
		log_action_prob = K.log(action_prob)

		loss = -log_action_prob*discount_reward_placeholder
		loss = K.mean(loss)

		adam = optimizers.Adam()

		updates = adam.get_updates(params=self.model.trainable_weights, loss=loss)

		self.train_function = K.function(inputs=[self.model.input,
												 action_onehot_placeholer,
												 discount_reward_placeholder],
										 outputs=[],
										 updates=updates)


	def get_action(self, state, invalid_moves=[], exploitation=1):

		shape = state.shape

		assert shape == self.input_dim, 'shape of state does not match network input layer'

		action_prob = np.squeeze(self.model.predict(np.expand_dims(state,0)))

		while 0 in action_prob:
			if np.sum(action_prob) == 0:
				action_prob = np.random.rand(4)
			else: 
				for index in np.where(action_prob == 0)[0]:
					action_prob[index] = np.min(np.array([value for value in action_prob if value != 0]))/2

		for move in invalid_moves:
			action_prob[move] = 0

		action_prob /= np.sum(action_prob)

		assert len(action_prob) == self.output_dim, 'Action probability shape does not match that of the network output layer'

		try:
			explore = np.random.choice(np.arange(self.output_dim), p=action_prob)
			exploit = np.argmax(action_prob)
			return np.random.choice(np.array([exploit, explore]), p=np.array([exploitation, 1-exploitation]))

		except:
			rand = np.random.rand(4)
			action_prob = rand/np.sum(rand)
			explore = np.random.choice(np.arange(self.output_dim), p=action_prob)
			exploit =  np.argmax(action_prob)
			return np.random.choice(np.array([exploit, explore]), p=np.array([exploitation, 1-exploitation]))

	def fit(self, S, A, R):

		action_onehot = np_utils.to_categorical(A, num_classes=self.output_dim)
		discount_reward = compute_discounted_R(R,self.discount_rate)

		assert S.shape[1:] == self.input_dim, 'Shape of state does not match network input layer'
		assert S.shape[0] == action_onehot.shape[0], 'Number of examples in state does not match number of examples in action'
		assert action_onehot.shape[1] == self.output_dim, 'Shape of action {} does not match that of the network output layer {}'.format(action_onehot.shape[1:],self.output_dim)

		assert len(discount_reward.shape) == 1, 'discounted reward should be a one-dimensional array but has {} dimension'.format(len(discount_reward))

		self.train_function([S, action_onehot, discount_reward])


def compute_discounted_R(R, discount_rate):

	discounted_r = np.zeros_like(R, dtype=np.float32)
	
	running_add = 0
	for t in reversed(range(len(R))):

		running_add = running_add * discount_rate + R[t]
		discounted_r[t] = running_add

	discounted_r -= ( discounted_r.mean() / discounted_r.std() )

	return discounted_r

