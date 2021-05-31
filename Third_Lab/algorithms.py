import numpy as np
import sys
from copy import deepcopy
from gym.wrappers import Monitor

from shedulers import get_epsilon_rate, get_learning_rate
from policies import select_action


class QLearning:
    def __init__(self, env, num_episodes, min_epsilon_rate, min_learning_rate, learning_rate,
                 epsilon_rate, gamma, max_win_streak=100, make_action_proba=1, learning_rate_is_constant=False,
                 epsilon_rate_is_constant=False, debug=True, render_maze=True, recording_folder='./videos',
                 enable_recording=True
                 ):

        self._env = env
        self._num_episodes = num_episodes
        self._min_epsilon_rate = min_epsilon_rate
        self._min_learning_rate = min_learning_rate
        self._learning_rate = learning_rate
        self._epsilon_rate = epsilon_rate
        self._gamma = gamma
        self._max_win_streak = max_win_streak
        self._make_action_proba = make_action_proba
        self._learning_rate_is_constant = learning_rate_is_constant
        self._epsilon_rate_is_constant = epsilon_rate_is_constant
        self._debug = debug
        self._render_maze = render_maze
        self._recording_folder = recording_folder
        self._enable_recording = enable_recording

    def get_simulation_parameters(self):

        maze_size = tuple((self._env.observation_space.high + np.ones(self._env.observation_space.shape)).astype(int))
        num_buckets = maze_size
        num_actions = self._env.action_space.n

        state_bounds = list(zip(self._env.observation_space.low, self._env.observation_space.high))

        decay_rate = np.prod(maze_size, dtype=float) / 10.0

        max_t = np.prod(maze_size, dtype=int) * 100
        solved_t = np.prod(maze_size, dtype=int)

        return num_buckets, num_actions, state_bounds, decay_rate, max_t, solved_t

    @staticmethod
    def state_to_bucket(state, state_bounds, num_buckets):
        bucket_indice = []
        for i in range(len(state)):
            if state[i] <= state_bounds[i][0]:
                bucket_index = 0
            elif state[i] >= state_bounds[i][1]:
                bucket_index = num_buckets[i] - 1
            else:
                bound_width = state_bounds[i][1] - state_bounds[i][0]
                offset = (num_buckets[i] - 1) * state_bounds[i][0] / bound_width
                scaling = (num_buckets[i] - 1) / bound_width
                bucket_index = int(round(scaling * state[i] - offset))
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)

    def simulate(self):

        total_rewards = []
        total_streaks = []
        steps_qty = []

        num_buckets, num_actions, state_bounds, decay_rate, max_t, solved_t = self.get_simulation_parameters()

        q_table = np.zeros(num_buckets + (num_actions,), dtype=float)

        if not self._epsilon_rate_is_constant:
            epsilon_rate_setup = get_epsilon_rate(0, self._min_epsilon_rate, decay_rate)
        else:
            epsilon_rate_setup = self._epsilon_rate

        if not self._learning_rate_is_constant:
            learning_rate_setup = get_learning_rate(0, self._min_learning_rate, decay_rate)
        else:
            learning_rate_setup = self._learning_rate

        num_streaks = 0

        self._env.render()

        for episode in range(self._num_episodes):
            # Reset the environment
            obv = self._env.reset()
            # the initial state
            state_0 = self.state_to_bucket(obv, state_bounds, num_buckets)
            total_reward = 0

            tmp_steps = []

            for t in range(max_t):
                # Select an action
                action = select_action(self._env, q_table, state_0, epsilon_rate_setup)
                # execute the action
                if np.random.choice([True, False], p=[self._make_action_proba, 1 - self._make_action_proba]):
                    obv, reward, done, _ = self._env.step(action)
                else:
                    action = self._env.action_space.sample()
                    obv, reward, done, _ = self._env.step(action)

                # Observe the result
                state = self.state_to_bucket(obv, state_bounds, num_buckets)
                total_reward += reward

                # Update the Q based on the result
                best_q = np.amax(q_table[state])
                q_table[state_0 + (action,)] += learning_rate_setup * (
                        reward + self._gamma * (best_q) - q_table[state_0 + (action,)])

                # Setting up for the next iteration
                state_0 = state

                tmp_steps.append(t)

                if self._debug:
                    if done or t >= max_t - 1:
                        print("\nEpisode = %d" % episode)
                        print("t = %d" % t)
                        print("Explore rate: %f" % epsilon_rate_setup)
                        print("Learning rate: %f" % learning_rate_setup)
                        print("Streaks: %d" % num_streaks)
                        print("Total reward: %f" % total_reward)
                        print("Average Streaks: %f" % (num_streaks / (episode + 1)))
                        print("Average Rewards: %f" % (total_reward / (episode + 1)))
                        print("")

                if self._render_maze:
                    self._env.render()

                if self._env.is_game_over():
                    sys.exit()
                if done:
                    print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                          % (episode, t, total_reward, num_streaks))
                    if t <= solved_t:
                        num_streaks += 1
                    else:
                        num_streaks = 0
                    break
                elif t >= max_t - 1:
                    print("Episode %d timed out at %d with total reward = %f."
                          % (episode, t, total_reward))

            # It's considered done when it's solved over 120 times consecutively
            if num_streaks > self._max_win_streak:
                break

            if not self._epsilon_rate_is_constant:
                epsilon_rate_setup = get_epsilon_rate(episode, self._min_epsilon_rate, decay_rate)
            if not self._learning_rate_is_constant:
                learning_rate_setup = get_learning_rate(episode, self._min_learning_rate, decay_rate)

            total_rewards.append(total_reward)
            total_streaks.append(num_streaks)
            steps_qty.append(tmp_steps[-1])

        return total_rewards, total_streaks, steps_qty

    def main(self):

        self._env = Monitor(self._env, self._recording_folder, video_callable=lambda episode: True, force=True)

        if self._enable_recording:
            self._env._start(self._recording_folder, video_callable=lambda episode: True, force=True)

        total_rewards, total_streaks, steps_qty = self.simulate()

        return total_rewards, total_streaks, steps_qty


class ValueIteration:
    def __init__(self, env, gamma, epsilon, make_action_proba, max_win_streak=100, debug=True, render_maze=True,
                 recording_folder='./videos', enable_recording=True):
        self._env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.make_action_proba = make_action_proba
        self._max_win_streak = max_win_streak
        self._debug = debug
        self._render_maze = render_maze
        self._recording_folder = recording_folder
        self._enable_recording = enable_recording
        self.v = {}
        self.policy = {}

        all_states = [(x, y) for x in range(self._env.maze_size[0]) for y in range(self._env.maze_size[0])]

        for s in all_states:
            self.policy[s] = np.random.choice(self._env.get_all_possible_actions(s))

        self.all_possible_actions = {state: self._env.get_all_possible_actions(state) for state in all_states}

        for s in all_states:
            reward = self._env.calculate_reward(s)
            self.v[s] = reward

    def get_simulation_parameters(self):

        maze_size = tuple((self._env.observation_space.high + np.ones(self._env.observation_space.shape)).astype(int))
        num_buckets = maze_size
        num_actions = self._env.action_space.n

        state_bounds = list(zip(self._env.observation_space.low, self._env.observation_space.high))

        decay_rate = np.prod(maze_size, dtype=float) / 10.0

        max_t = np.prod(maze_size, dtype=int) * 100
        solved_t = np.prod(maze_size, dtype=int)

        return num_buckets, num_actions, state_bounds, decay_rate, max_t, solved_t

    @staticmethod
    def state_to_bucket(state, state_bounds, num_buckets):
        bucket_indice = []
        for i in range(len(state)):
            if state[i] <= state_bounds[i][0]:
                bucket_index = 0
            elif state[i] >= state_bounds[i][1]:
                bucket_index = num_buckets[i] - 1
            else:
                bound_width = state_bounds[i][1] - state_bounds[i][0]
                offset = (num_buckets[i] - 1) * state_bounds[i][0] / bound_width
                scaling = (num_buckets[i] - 1) / bound_width
                bucket_index = int(round(scaling * state[i] - offset))
            bucket_indice.append(bucket_index)
        return tuple(bucket_indice)

    def simulate(self):

        self._env.render()
        num_buckets, num_actions, state_bounds, decay_rate, max_t, solved_t = self.get_simulation_parameters()

        total_rewards = []
        total_streaks = []
        steps_qty = []

        num_streaks = 0
        total_reward = 0

        tmp_steps = []

        iteration = 0
        while iteration < max_t:
            best_chance = 0
            all_states = [(x, y) for x in range(self._env.maze_size[0]) for y in range(self._env.maze_size[0])]
            for state in all_states:
                if not self._env.check_done(state):
                    all_portals_locations = [[location for location in portal.locations]
                                             for portal in self._env.maze_view.maze.portals]

                    portal_coordinates = None
                    for portal_locations in all_portals_locations:
                        for location in portal_locations:
                            if np.array_equal(state, location):
                                portal_coordinates = portal_locations[1 - portal_locations.index(location)]

                    if portal_coordinates is not None:
                        state = portal_coordinates

                    all_possible_actions = self.all_possible_actions.get(state)
                    if not len(all_possible_actions):
                        continue
                    old_value = deepcopy(self.v[state])

                    temp_values = []

                    for action in all_possible_actions:
                        coord = state + np.array(self._env.maze_view.maze.STEPS[action])

                        value = self.v[tuple(coord)]

                        if len(all_possible_actions) == 1:
                            main_prob = 1
                            additional_prob = 0
                        else:
                            main_prob = self.make_action_proba
                            additional_prob = (1 - self.make_action_proba) / (len(all_possible_actions) - 1)

                        add_value_sum = 0
                        for another_action in all_possible_actions:
                            if another_action != action:
                                another_coord = state + np.array(self._env.maze_view.maze.STEPS[another_action])
                                add_value_sum += self.v[tuple(another_coord)]

                        temp_values.append(value * main_prob + additional_prob * add_value_sum)

                    self.v[state] = self._env.calculate_reward(state) + self.gamma * np.max(temp_values)
                    self.policy[state] = all_possible_actions[np.argmax(temp_values)]

                    best_chance = max(best_chance, np.abs(old_value - self.v[state]))

            if best_chance < self.epsilon:
                break
            iteration += 1

            try:
                state = self._env.reset()
            except:
                state = np.zeros(2)

            for t in range(50):
                # Select an action
                action = self.policy[tuple(state)]
                # execute the action
                if np.random.choice([True, False], p=[self.make_action_proba, 1 - self.make_action_proba]):
                    state, reward, done, _ = self._env.step(action)
                    reward = self.v[tuple(state)]
                else:
                    action = self._env.action_space.sample()
                    state, reward, done, _ = self._env.step(action)
                    reward = self.v[tuple(state)]
                # Observe the result
                total_reward += reward

                tmp_steps.append(t)

                if self._debug:
                    if done or t >= max_t - 1:
                        print('Iteration: {}'.format(iteration))
                        print("t = %d" % t)
                        print("Streaks: %d" % num_streaks)
                        print("Total reward: %f" % total_reward)
                        print("Average Streaks: %f" % (num_streaks / (t + 1)))
                        print("Average Rewards: %f" % (total_reward / (t + 1)))
                        print("")

                if self._render_maze:
                    self._env.render()

                if self._env.is_game_over():
                    sys.exit()
                if done:
                    print("Finished after %f time steps with total reward = %f (streak %d)."
                          % (t, total_reward, num_streaks))
                    if t <= solved_t:
                        num_streaks += 1
                    else:
                        num_streaks = 0
                    break
                elif t >= max_t - 1:
                    print("Timed out at %d with total reward = %f."
                          % (t, total_reward))

            # It's considered done when it's solved over 120 times consecutively
                if num_streaks > self._max_win_streak:
                    print('BREAK')
                    break

                total_rewards.append(total_reward)
                total_streaks.append(num_streaks)
                steps_qty.append(tmp_steps[-1])

        return total_rewards, total_streaks, steps_qty

    def main(self):
        self._env = Monitor(self._env, self._recording_folder, video_callable=lambda episode: True, force=True)

        if self._enable_recording:
            self._env._start(self._recording_folder, video_callable=lambda episode: True, force=True)

        total_rewards, total_streaks, steps_qty = self.simulate()

        return total_rewards, total_streaks, steps_qty

