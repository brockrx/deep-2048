import random, math, json
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from logic import Game
from keras import optimizers, layers
from keras import utils as np_utils
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling1D, Add, Concatenate

class Agent():

	def __init__(self, input_dim=[4,4,1], output_dim=4, loss_penalty=50, invalid_move_penalty=0, discount_rate=0.995, conv_hidden_dims=[128]*3, dense_dim_factor=64, reward_decay=0.5):
		
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.conv_hidden_dims = conv_hidden_dims
		self.dense_dim_factor = dense_dim_factor
		self.dense_hidden_dims = [128]
		self.invalid_move_penalty = invalid_move_penalty
		self.loss_penalty = loss_penalty
		self.discount_rate = discount_rate
		self.tile_list = [32, 64, 128, 256, 512, 1024, 2048]
		self.reward_dic = [x for x in range(11)]
		self.reward_decay = reward_decay


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

		#convolutional_layers = [Conv2D(dim, 2, strides=1, padding='same') for dim in self.conv_hidden_dims]
		#dense_layers = [Dense(dim) for dim in self.dense_hidden_dims]

		#network_dic = {}
		#self.input_dic = {}

		#for board in range(8):

		#	self.input_dic[board] = layers.Input(shape=list(self.input_dim))
		#	net = self.input_dic[board]

		#	for layer in convolutional_layers:
		#		net = layer(net)
		#		net = Activation("sigmoid")(net)

		#	net = Flatten()(net)

		#	for layer in dense_layers:
		#		net = layer(net)
		#		net = Activation("sigmoid")(net)

		#	net = Dense(self.output_dim)(net)
		#	net = Activation("softmax")(net)

		#	network_dic[board] = net

		#net = Add()([network_dic[net] for net in network_dic.keys()])

		#net = Dense(self.output_dim)(net)
		#net = Activation('softmax')(net)

		#self.model = Model(inputs=[self.input_dic[key] for key in self.input_dic.keys()], outputs=net)
		
		self.X = layers.Input(shape=self.input_dim)
		net = self.X

		net_path_one = Conv2d(128)

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

		if type(state) == list:
			shape = state[0].shape[1:]

		elif type(state) == np.ndarray:
			shape = state.shape[1:]

		assert shape == self.input_dim, 'shape of state does not match network input layer'

		action_prob = np.squeeze(self.model.predict(state))

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

		assert S[0].shape == self.input_dim, 'Shape of state does not match network input layer'
		assert S.shape[0] == action_onehot.shape[0], 'Number of examples in state does not match number of examples in action'
		assert action_onehot.shape[1] == self.output_dim, 'Shape of action {} does not match that of the network output layer {}'.format(action_onehot.shape[1:],self.output_dim)

		assert len(discount_reward.shape) == 1, 'discounted reward should be a one-dimensional array but has {} dimension'.format(len(discount_reward))
		
		self.train_function([S, action_onehot, discount_reward])

	def run_episode(self):

		game = Game()

		state_list = []
		action_list = []
		reward_list = []

		previous_highest_tile = game.highest_tile()

		exploitation = self.initial_exploitation
		exploitation_decay = self.initial_exploitation/self.average_game_length*0.8

		done = False

		while not done:

			state = game.get_state()

			action = self.get_action(state)

			invalid_moves = []

			while not game.check_valid_move(action):

				invalid_moves.append(action)
				action = self.get_action(state, invalid_moves=invalid_moves)

			_, _, done, tiles_made = game.step(action)

			reward = sum([self.reward_dic[int(math.log(i,2))]*int(math.log(i,2)) for i in tiles_made])
			self.update_reward_dic(tiles_made)

			highest_tile = game.highest_tile()

			new_state = game.get_state()

			state_list.append(state[0,:,:,:])
			action_list.append(int(action))
			reward_list.append(reward)

			previous_highest_tile = highest_tile

			exploitation = np.max([0,exploitation-exploitation_decay])

		state_list.append(state[0,:,:,:])
		action_list.append(int(action))
		reward_list.append(-self.loss_penalty)
		total_reward = sum(reward_list)
		score = game.score

		self.average_game_length = self.average_game_length*0.1 + len(state_list)*0.9

		return (total_reward, score, highest_tile, action_list, state_list, reward_list)

	def update_reward_dic(self, tiles_made):
		for tile in tiles_made:
			index = int(math.log(tile,2))
			self.reward_dic[index] *= self.reward_decay


	def train(self, episodes):

		self.save_dic = {}
		self.save_dic['reward_list'] = []
		self.save_dic['score_list'] = []
		self.save_dic['highest_tile_list'] = []
		self.save_dic['state_list'] = []

		self.train_reward_list = []
		self.train_score_list = []
		self.train_highest_tile_list = []
		self.train_highest_tile_dic = {}
		self.train_episode_list = []

		self.create_graphs()

		self.initial_exploitation = 0
		self.average_game_length = 50

		for episode in range(episodes):

			total_reward, score, highest_tile, action_list, state_list, reward_list = self.run_episode()

			self.reward_dic = [x for x in range(11)]

			state = np.array(state_list)
			action = np.array(action_list)
			reward = np.array(reward_list)

			self.fit(state, action, reward)

			self.train_reward_list.append(total_reward)
			self.train_score_list.append(score)
			self.train_highest_tile_list.append(highest_tile)
			self.train_episode_list.append(episode)

			self.update_graphs()

			self.save_dic['reward_list'].append(self.train_reward_list)
			self.save_dic['score_list'].append(self.train_score_list)
			self.save_dic['highest_tile_list'].append(self.train_highest_tile_list)
			self.save_dic['state_list'].append([state.tolist() for state in state_list])

			if (episode+1)%100 == 0:
				with open('train_history.csv', 'w') as file:
					json.dump(self.save_dic, file)

			self.initial_exploitation = min([self.initial_exploitation+0.0001, 0.9])

	def create_graphs(self):

		for tile in self.tile_list:
			self.train_highest_tile_dic[tile] = []
		self.train_reward_plot = []
		self.train_score_plot = []

		self.tile_figure = plt.figure()
		self.tile_ax = self.tile_figure.add_subplot(111)
		plt.ylim(0,1)
		self.tile_ax.legend()
		self.tile_figure.show()
		self.tile_figure.canvas.draw()

		self.reward_figure = plt.figure()
		self.reward_ax = self.reward_figure.add_subplot(111)
		self.reward_ax.legend()
		self.reward_figure.show()
		self.reward_figure.canvas.draw()

		self.score_figure = plt.figure()
		self.score_ax = self.score_figure.add_subplot(111)
		self.score_ax.legend()
		self.score_figure.show()
		self.score_figure.canvas.draw()

	def update_graphs(self):

		self.tile_ax.clear()
		for tile in self.tile_list:
			self.train_highest_tile_dic[tile].append(self.train_highest_tile_list[-500:].count(tile)/min([500,len(self.train_highest_tile_list)]))
			self.tile_ax.plot(self.train_highest_tile_dic[tile], label=str(tile))
		self.tile_ax.legend(loc='upper left')
		self.tile_figure.canvas.draw()

		self.train_reward_plot.append(sum(self.train_reward_list[-500:])/min([500,len(self.train_reward_list)]))
		self.reward_ax.clear()
		self.reward_ax.plot(self.train_reward_plot, label='Reward')
		self.reward_ax.legend(loc='upper left')
		self.reward_figure.canvas.draw()

		self.train_score_plot.append(sum(self.train_score_list[-500:])/min([500,len(self.train_score_list)]))
		self.score_ax.clear()
		self.score_ax.plot(self.train_score_plot, label='Score')
		self.score_ax.legend(loc='upper left')
		self.score_figure.canvas.draw()
		

def compute_discounted_R(R, discount_rate):

	discounted_r = np.zeros_like(R, dtype=np.float32)
	
	running_add = 0
	for t in reversed(range(len(R))):

		running_add = running_add * discount_rate + R[t]
		discounted_r[t] = running_add

	discounted_r -= ( discounted_r.mean() / discounted_r.std() )

	return discounted_r