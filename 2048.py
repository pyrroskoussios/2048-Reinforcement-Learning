import copy
import random
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from tensorflow import keras

class AgentParameters():

	def __init__(
			self,
			max_memory_size,
			min_memory_size,
			batch_size,
			learning_rate
		):

		self.max_memory_size = max_memory_size
		self.min_memory_size = min_memory_size
		self.batch_size = batch_size
		self.learning_rate = learning_rate


class TrainerParameters():

	def __init__(
			self,
			save_model,
			save_model_every,
			train_model_every,
			update_target_every, 
			episodes,
			epsilon_start, 
			epsilon_decay, 
			epsilon_min, 
			gamma,
		):

		self.save_model = save_model
		self.save_model_every = save_model_every
		self.train_model_every = train_model_every
		self.update_target_every = update_target_every
		self.episodes = episodes
		self.epsilon_start = epsilon_start
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.gamma = gamma


class Environment:

	def __init__(self):
		self.reset()

	def _sumCol(self, col_set):
		for j in range(len(col_set)-1):
			if col_set[j] == col_set[j+1]:
				col_set.pop(j)
				col_set[j] *= 2
				col_set.append(0)
				self.score += col_set[j]

		col_set.extend([0] * (4 - len(col_set)))

	def reset(self):
		self.grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
		self.last_grid = self.grid
		self.score = 0
		self.last_score = self.score
		self.empty_tiles = list(range(16))

		for i in range(2):
			tile_to_fill = random.choice(self.empty_tiles)
			number_prob = random.random()

			if number_prob > 0.9:
				self.grid[int(tile_to_fill/4)][tile_to_fill%4] = 4
			else:
				self.grid[int(tile_to_fill/4)][tile_to_fill%4] = 2

			self.empty_tiles.remove(tile_to_fill)

	def step(self, action=None, training_mode=False):
		self.last_grid = copy.deepcopy(self.grid)
		self.last_score = self.score

		if action == None:
			action = input("Direction: ")

		if action in ["w", "W", "s", "S"]:
			for i in range(0, 4):  
				col = [self.grid[0][i], self.grid[1][i], self.grid[2][i], self.grid[3][i]]
				col_set = [x for x in col if x]

				if action in ["w", "W"]:
					self._sumCol(col_set)
				else:
					col_set.reverse()
					self._sumCol(col_set)
					col_set.reverse()

				for j in range(4):
					self.grid[j][i] = col_set[j]

		elif action in ["a", "A", "d", "D"]:
			for i in range(0, 4):
				col = [self.grid[i][0], self.grid[i][1], self.grid[i][2], self.grid[i][3]]
				col_set = [x for x in col if x]

				if action in ["a", "A"]:
					self._sumCol(col_set)
				else:
					col_set.reverse()
					self._sumCol(col_set)
					col_set.reverse()

				for j in range(4):
					self.grid[i][j] = col_set[j]

		if self.last_grid == self.grid or action not in ["w", "W", "s", "S", "a", "A", "d", "D"]:
			if not training_mode:
				self.step()
			return 0

		self.empty_tiles = []
		for i in range(16):
			if not self.grid[int(i/4)][i%4]:
				self.empty_tiles.append(i)

		tile_to_fill = random.choice(self.empty_tiles)
		number_prob = random.random()

		if number_prob > 0.9:
			self.grid[int(tile_to_fill/4)][tile_to_fill%4] = 4
		else:
			self.grid[int(tile_to_fill/4)][tile_to_fill%4] = 2

		self.empty_tiles.remove(tile_to_fill)

		return 1

	def print(self):
		string_grid = copy.deepcopy(self.grid)
		string_grid = [str(y) if y else " " for x in string_grid for y in x]
		
		string = "+---------+---------+---------+---------+" + "\n"
		for i in range(0, 4):
			string += "|         |         |         |         |" + "\n"
			string += "| " + string_grid[0+i*4].center(7)  + " | " + string_grid[1+i*4].center(7)  + " | " + string_grid[2+i*4].center(7)  + " | " + string_grid[3+i*4].center(7)  + " |" + "\n"
			string += "|         |         |         |         |" + "\n"
			string += "+---------+---------+---------+---------+" + "\n"

		print(string, end="")


class Agent:

	def __init__(self, agent_parameters):
		self.agent_parameters = agent_parameters

		self.replay_memory = deque(maxlen=agent_parameters.max_memory_size)

		self.predict_network, self.target_network = self._initializeNetworks()

	def _encodeGrids(self, grids):
		return [[[[[int(bit) for bit in str_bit] for str_bit in format(tile, '017b')[:-1]] for tile in row] for row in grid] for grid in grids]

	def _buildNetowrk(self):
		input_shape = (4, 4, 16, 1)

		inputs = keras.Input(shape=input_shape, batch_size=None)
		conv1 = keras.layers.Conv3D(32, (3, 3, 3), padding="same", activation="relu")(inputs)
		conv2 = keras.layers.Conv3D(32, (3, 3, 3), activation="relu")(conv1)
		flatten = keras.layers.Flatten()(conv2)
		dense1 = keras.layers.Dense(16, activation="relu")(flatten)
		outputs = keras.layers.Dense(4, activation="linear")(dense1)

		network = keras.Model(inputs=inputs, outputs=outputs)
		network.compile(optimizer=keras.optimizers.Adam(learning_rate=self.agent_parameters.learning_rate, beta_1=0.9, beta_2=0.999), loss="mse")

		return network

	def _initializeNetworks(self):
		predict_network = self._buildNetowrk()
		target_network = self._buildNetowrk()

		target_network.set_weights(predict_network.get_weights())

		return predict_network, target_network

	def _getPossibleActions(self, environment):
		possible_actions = []

		for action in ["w", "s", "a", "d"]:
			env_copy = copy.deepcopy(environment)

			if env_copy.step(action, True):
				possible_actions.append(action)

		return possible_actions

	def _predictOnGrids(self, network, grids):
		return network.predict_on_batch(np.array(self._encodeGrids(grids)))

	def getAction(self, environment, epsilon):
		index_to_action = {0:"w", 1:"a", 2:"s", 3:"d"}

		possible_actions = self._getPossibleActions(environment)

		exploit = random.random() > epsilon

		if exploit:
			predictions = list(self._predictOnGrids(self.predict_network, [environment.grid]).flatten())
			masked = [predictions[ix] if index_to_action[ix] in possible_actions else float("-inf") for ix in range(4)]
			return index_to_action[max(enumerate(masked), key=lambda x: x[1])[0]]
		else:
			return random.choice(possible_actions)

	def getReward(self, environment):
		return (environment.score - environment.last_score) / 10

	def getDoneState(self, environment):
		return not bool(len(self._getPossibleActions(environment)))

	def getGridState(self, environment):
		return copy.deepcopy(environment.grid)

	def updateReplayMemory(self, current_state, action, reward, next_state, done):
		self.replay_memory.append((current_state, action, reward, next_state, done))

	def updateTargetNetwork(self):
		self.target_network.set_weights(self.predict_network.get_weights())

	def updatePredictNetwork(self, gamma):
		if len(self.replay_memory) < self.agent_parameters.min_memory_size:
			return

		action_to_index = {"w":0, "a":1, "s":2, "d":3}

		batch = random.sample(self.replay_memory, self.agent_parameters.batch_size)

		current_states = [memory[0] for memory in batch]
		actions = [memory[1] for memory in batch]
		rewards = [memory[2] for memory in batch]
		next_states = [memory[3] for memory in batch]
		dones = [memory[4] for memory in batch]

		predicted_current_states = self._predictOnGrids(self.predict_network, current_states)
		predicted_next_states = self._predictOnGrids(self.target_network, next_states)

		new_values = [rewards[i] if dones[i] else rewards[i] + gamma * np.max(predicted_next_states[i]) for i in range(self.agent_parameters.batch_size)]

		loss_values = [(new_values[i] - predicted_next_states[i][action_to_index[actions[i]]]) ** 2 for i in range(self.agent_parameters.batch_size)]

		for i in range(self.agent_parameters.batch_size):
			predicted_current_states[i][action_to_index[actions[i]]] = new_values[i]

		self.predict_network.fit(np.array(self._encodeGrids(current_states)), predicted_current_states, epochs=1, verbose=0, batch_size=self.agent_parameters.batch_size, shuffle=False) 

		return sum(loss_values) / self.agent_parameters.batch_size

	def saveModel(self, training_loss, training_score, average_window):
		self.predict_network.save("/Users/pyrrosk/Dropbox/Coding/2048/models/predict_network", save_format="h5")

		averaged_loss = np.cumsum(training_loss, dtype=float)
		averaged_loss[average_window:] = averaged_loss[average_window:] - averaged_loss[:-average_window]
		averaged_loss = np.concatenate(([None for i in range(average_window)], averaged_loss[average_window - 1:] / average_window))

		averaged_score = np.cumsum(training_score, dtype=float)
		averaged_score[average_window:] = averaged_score[average_window:] - averaged_score[:-average_window]
		averaged_score = np.concatenate(([None for i in range(average_window)], averaged_score[average_window - 1:] / average_window))

		loss_plot = plt.figure()
		plt.plot(training_loss, color="lightblue")
		plt.plot(averaged_loss, color="slateblue")
		plt.title("DQN Loss")
		plt.ylabel("Loss Value")
		plt.xlabel("Episode")

		score_plot = plt.figure()
		plt.plot(training_score, color="lightsalmon")
		plt.plot(averaged_score, color="orangered")
		plt.title("Score")
		plt.ylabel("Reward")
		plt.xlabel("Episode")

		loss_plot.savefig("/Users/pyrrosk/Dropbox/Coding/2048/models/loss_plot", dpi=200)
		score_plot.savefig("/Users/pyrrosk/Dropbox/Coding/2048/models/score_plot", dpi=200)


class Trainer():

	def train(self, agent_parameters, trainer_parameters):
		agent = Agent(agent_parameters)
		env = Environment()

		epsilon = trainer_parameters.epsilon_start

		training_loss = []
		training_score = []

		for episode in range(trainer_parameters.episodes):

			env.reset()

			if trainer_parameters.save_model and episode and not episode % trainer_parameters.save_model_every:
				agent.saveModel(training_loss, training_score, trainer_parameters.save_model_every)

			if not episode % trainer_parameters.update_target_every:
				agent.updateTargetNetwork()

			done = False
			move_num = 0
			game_loss = []

			while not done:
				current_state = agent.getGridState(env)
				action = agent.getAction(env, epsilon)

				assert env.step(action, True), "Invalid Action!"

				reward = agent.getReward(env)
				next_state = agent.getGridState(env)
				done = agent.getDoneState(env)

				agent.updateReplayMemory(current_state, action, reward, next_state, done)

				if not move_num % trainer_parameters.train_model_every:
					loss = agent.updatePredictNetwork(trainer_parameters.gamma)
					if loss is not None:
						game_loss.append(loss)

				move_num += 1

			episode_loss = sum(game_loss) / len(game_loss) if game_loss else 0

			print("Episode: ", episode, " Loss: ", episode_loss, " Epsilon: ", epsilon, " Moves: ", move_num, " Score: ", env.score)

			training_loss.append(episode_loss)
			training_score.append(env.score)

			epsilon = trainer_parameters.epsilon_min if epsilon * trainer_parameters.epsilon_decay < trainer_parameters.epsilon_min else epsilon * trainer_parameters.epsilon_decay

	def gridSearch(self):
		max_memory_sizes = [5000, 25000]
		learning_rates = [0.0001, 0.00001, 0.000001]
		batch_sizes = [16, 32, 64]
		update_target_everys = [1000, 5000, 10000]
		epsilon_decays = [0.99, 0.999]
		gammas = [0.75, 0.9, 0.99]

		params = []
		for max_memory_size in max_memory_sizes:
			for learning_rate in learning_rates:
				for batch_size in batch_sizes:
					for update_target_every in update_target_everys:
						for epsilon_decay in epsilon_decays:
							for gamma in gammas:
								params.append((AgentParameters(max_memory_size, 1000, batch_size, learning_rate), TrainerParameters(True, 50, 4, update_target_every, 15000, 1, epsilon_decay, 0.1, gamma)))

		for param in params:
			self.train(param[0], param[1])

		


if __name__ == "__main__":
	t_params = TrainerParameters(False, 50, 4, 9000, 10000, 1, 0.999, 0.1, 0.9)
	a_params = AgentParameters(5000, 1000, 64, 0.00001)

	trainer = Trainer()

	trainer.train(a_params, t_params)

	"""env = Environment()
	env.print()
	while True:
		env.step()
		env.print()"""

