import math, random
import numpy as np

class Game():

	index = {0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3), 4: (1,0), 5: (1,1), 6: (1,2), 7: (1,3), 8: (2,0), 9: (2,1), 10: (2,2), 11: (2,3), 12: (3,0), 13: (3,1), 14: (3,2), 15: (3,3)}

	def __init__(self):
		self.board = np.full((4,4), np.nan)
		self.initialize_game()
		self.score = 0
		self.game_over = False
		self.direction_dic = {0: 'left', 1: 'right', 2: 'up', 3: 'down'}
		return None		

	def __str__(self):
		return str(self.board)


	def add_spawn(self, index=None, number=None):
		if index== None:
			location = random.randint(0, 15)	
			index = self.index[location]
			while not np.isnan(self.board[index[0],index[1]]):
				location = random.randint(0, 15)	
				index = self.index[location]
		if number == None:
			number = 2 if random.uniform(0,1) < 0.8 else 4
		self.board[index[0]][index[1]] = number

	def initialize_game(self):
		self.add_spawn()
		self.add_spawn()

	def orient_board(self, direction, order):
		if order == 'enter':
			if direction == 'up' or direction == 'down':
				self.board = self.board.T

			if direction == 'right' or direction == 'down':
				self.board = np.flip(self.board, 1)

		if order == 'exit':

			if direction == 'right' or direction == 'down':
				self.board = np.flip(self.board, 1)

			if direction == 'up' or direction == 'down':
				self.board = self.board.T

	def swipe(self, direction, spawn=True):
		# orient board
		if self.check_valid_move(direction):
			self.orient_board(direction, 'enter')
			M, N = self.board.shape
			for m in range(M):
				added = [False]*4
				for n in range(1,N):
					if np.isnan(self.board[m,n]):
						continue
					else:
						for i in reversed(range(n)):
							if np.isnan(self.board[m,i]):
								if i == 0:
									self.board[m,i] = self.board[m,n]
									self.board[m,n] = np.nan
									break

							else:
								if self.board[m,i] == self.board[m,n] and not added[i]:
									self.score += int(self.board[m,n] + self.board[m,i])
									self.board[m,i] = int(self.board[m,n] + self.board[m,i])
									self.board[m,n] = np.nan
									added[i] = True
									break

								elif np.isnan(self.board[m,i+1]):
									self.board[m,i+1] = self.board[m,n]
									self.board[m,n] = np.nan
									break

								else:
									break

			self.orient_board(direction, 'exit')
			if spawn == True:
				self.add_spawn()

	def step(self, action):
		info = None
		direction = self.direction_dic[action]
		previous_score = self.score
		self.swipe(direction)
		done = self.check_loss()
		reward = self.score-previous_score
		observation = self.convert_board()
		return (observation, reward, done, info)

	def check_valid_move(self, direction):
		if type(direction) != str:
			direction = self.direction_dic[direction]
		self.orient_board(direction, 'enter')
		M, N = self.board.shape
		for m in range(M):
			for n in reversed(range(N)):
				if n != 0:
					if not np.isnan(self.board[m,n]) and np.isnan(self.board[m,n-1]) or self.board[m,n] == self.board[m,n-1]:
						self.orient_board(direction, 'exit')
						return True
		self.orient_board(direction, 'exit')
		return False

	def check_loss(self):
		M, N = self.board.shape
		for m in range(M):
			for n in range(N):
				if np.isnan(self.board[m,n]):
					return False

		for m in range(M):
			for n in range(N-1):
				if self.board[m,n] == self.board[m,n+1] or self.board.T[m,n] == self.board.T[m,n+1]:
					return False

		self.game_over = True
		return True

	def convert_board(self):
		M, N = self.board.shape
		board = np.zeros((M,N))
		for m in range(M):
			for n in range(N):
				if not np.isnan(self.board[m,n]):
					board[m,n] = np.log2(self.board[m,n])
		return board

	def read_prediction(self, prediction):
		return direction_dic[np.argmax(prediction)]

	def reset(self):
		self.__init__()

	def highest_tile(self):
		return 2**np.max(self.convert_board())

	def get_state(self, log=False):
		M, N = self.board.shape
		board = np.zeros((M,N))
		for m in range(M):
			for n in range(N):
				if not np.isnan(self.board[m,n]):
					if log == True:
						board[m,n] = int(np.log2(self.board[m,n]))
					else:
						board[m,n] = int(self.board[m,n])
		state = board
		state = np.concatenate((state, board.T), axis=1)
		state = np.concatenate((state, np.flip(board,1)), axis=1)
		state = np.concatenate((state, np.flip(board.T,1)), axis=1)
		state = np.concatenate((state, np.flip(board,0)), axis=1)
		state = np.concatenate((state, np.flip(board.T,0)), axis=1)
		state = np.concatenate((state, np.flip(np.flip(board,1),0)), axis=1)
		state = np.concatenate((state, np.flip(np.flip(board.T,1),0)), axis=1)
		
		return np.expand_dims(board,2)

	def board_list(self):
		return [[value if not np.isnan(value) else 0 for value in row] for row in self.board]