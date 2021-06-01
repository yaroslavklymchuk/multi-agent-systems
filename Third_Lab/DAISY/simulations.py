import numpy as np
import sys
from gym.wrappers import Monitor
from copy import deepcopy

from shedulers import SHEDULERS_MAPPING #get_explore_rate, get_learning_rate
from policies import select_action


def get_simulation_parameters(env):

    maze_size = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    num_buckets = maze_size
    num_actions = env.action_space.n

    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))

    max_t = np.prod(maze_size, dtype=int) * 100
    solved_t = np.prod(maze_size, dtype=int)

    return num_buckets, num_actions, state_bounds, max_t, solved_t


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


def simulate_q_learning(q_table, env, num_episodes, epsilon, lr, epsilon_shedule, lr_shedule, gamma, max_win_streak,
                        debug=True, render_maze=True
                        ):

    total_rewards = []
    total_streaks = []
    steps_qty = []

    num_buckets, num_actions, state_bounds, max_t, solved_t = get_simulation_parameters(env)

    if not epsilon_shedule:
        eps = epsilon
    else:
        epsilon_shedule_rule = list(epsilon_shedule.keys())[0]
        if epsilon_shedule_rule == 'smoothing_sheduler':
            smoothing_coefficient = epsilon_shedule.get(epsilon_shedule_rule)
            eps = SHEDULERS_MAPPING.get(epsilon_shedule_rule)(epsilon, smoothing_coefficient)
        else:
            decay = epsilon_shedule.get(epsilon_shedule_rule)
            eps = SHEDULERS_MAPPING.get(epsilon_shedule_rule)(0, epsilon, decay)

    if not lr_shedule:
        learning_rate = lr
    else:
        learning_rate_shedule_rule = list(lr_shedule.keys())[0]
        if learning_rate_shedule_rule == 'smoothing_sheduler':
            smoothing_coefficient = lr_shedule.get(learning_rate_shedule_rule)
            learning_rate = SHEDULERS_MAPPING.get(learning_rate_shedule_rule)(lr, smoothing_coefficient)
        else:
            decay = lr_shedule.get(learning_rate_shedule_rule)
            learning_rate = SHEDULERS_MAPPING.get(learning_rate_shedule_rule)(0, lr, decay)

    # if not explore_rate_is_constant:
    #     explore_rate_setup = get_explore_rate(0, min_explore_rate, decay_rate)
    # else:
    #     explore_rate_setup = explore_rate
    #
    # if not learning_rate_is_constant:
    #     learning_rate_setup = get_learning_rate(0, min_learning_rate, decay_rate)
    # else:
    #     learning_rate_setup = learning_rate

    num_streaks = 0

    env.render()

    for episode in range(num_episodes):
        # Reset the environment
        try:
            obv = env.reset()
        except:
            obv = np.zeros(2)
        # the initial state
        state_0 = state_to_bucket(obv, state_bounds, num_buckets)
        total_reward = 0

        tmp_steps = []

        for t in range(max_t):
            # Select an action
            action = select_action(env, q_table, state_0, eps)
            # execute the action
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv, state_bounds, num_buckets)
            total_reward += reward

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (
                    reward + gamma * (best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            tmp_steps.append(t)

            if debug:
                if done or t >= max_t - 1:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Explore rate: %f" % eps)
                    print("Learning rate: %f" % learning_rate)
                    print("Streaks: %d" % num_streaks)
                    print("Total reward: %f" % total_reward)
                    print("Average Streaks: %d" % (num_streaks / (episode + 1)))
                    print("Average Rewards: %d" % (total_reward / (episode + 1)))
                    print("")

            if render_maze:
                env.render()

            if env.is_game_over():
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
        if num_streaks > max_win_streak:
            break

        if epsilon_shedule:
            epsilon_shedule_rule = list(epsilon_shedule.keys())[0]
            if epsilon_shedule_rule == 'time_based_sheduler':
                smoothing_coefficient = epsilon_shedule.get(epsilon_shedule_rule)
                eps = SHEDULERS_MAPPING.get(epsilon_shedule_rule)(epsilon, smoothing_coefficient)
            else:
                decay = epsilon_shedule.get(epsilon_shedule_rule)
                eps = SHEDULERS_MAPPING.get(epsilon_shedule_rule)(episode, epsilon, decay)

        if lr_shedule:
            learning_rate_shedule_rule = list(lr_shedule.keys())[0]
            if learning_rate_shedule_rule == 'time_based_sheduler':
                smoothing_coefficient = lr_shedule.get(learning_rate_shedule_rule)
                learning_rate = SHEDULERS_MAPPING.get(learning_rate_shedule_rule)(learning_rate,
                                                                                  smoothing_coefficient)
            else:
                decay = lr_shedule.get(learning_rate_shedule_rule)
                learning_rate = SHEDULERS_MAPPING.get(learning_rate_shedule_rule)(episode, learning_rate, decay)

        # if not explore_rate_is_constant:
        #     explore_rate_setup = get_explore_rate(episode, min_explore_rate, decay_rate)
        # if not learning_rate_is_constant:
        #     learning_rate_setup = get_learning_rate(episode, min_learning_rate, decay_rate)

        total_rewards.append(total_reward)
        total_streaks.append(num_streaks)
        steps_qty.append(tmp_steps[-1])

    return total_rewards, total_streaks, steps_qty


def qlearning_main(env, num_episodes, epsilon, lr, epsilon_shedule, lr_shedule, gamma, max_win_streak,
                   debug=True, render_maze=True, recording_folder='./videos', enable_recording=True):

    env = Monitor(env, recording_folder, video_callable=lambda episode: True, force=True)
    num_buckets, num_actions, state_bounds, max_t, solved_t = get_simulation_parameters(env)

    q_table = np.zeros(num_buckets + (num_actions,), dtype=float)

    if enable_recording:
        env._start(recording_folder, video_callable=lambda episode: True, force=True)

    total_rewards, total_streaks, steps_qty = simulate_q_learning(env=env, q_table=q_table, num_episodes=num_episodes,
                                                                  epsilon=epsilon, lr=lr,
                                                                  epsilon_shedule=epsilon_shedule,
                                                                  lr_shedule=lr_shedule, gamma=gamma,
                                                                  max_win_streak=max_win_streak, debug=debug,
                                                                  render_maze=render_maze
                                                                  )

    return total_rewards, total_streaks, steps_qty


def simulate_value_iteration(env, gamma, epsilon, v, policy, all_possible_actions, make_action_proba, max_win_streak=100,
                             debug=True, render_maze=True):

    env.render()
    num_buckets, num_actions, state_bounds, max_t, solved_t = get_simulation_parameters(env)

    total_rewards = []
    total_streaks = []
    steps_qty = []

    num_streaks = 0
    total_reward = 0

    tmp_steps = []

    iteration = 0
    while iteration < max_t:
        best_chance = 0
        all_states = [(x, y) for x in range(env.maze_size[0]) for y in range(env.maze_size[0])]
        for state in all_states:
            if not env.check_done(state):
                possible_actions = all_possible_actions.get(state)
                if not len(possible_actions):
                    continue
                old_value = deepcopy(v[state])

                temp_values = []

                for action in possible_actions:
                    coord = state + np.array(env.maze_view.maze.STEPS[action])

                    value = v[tuple(coord)]

                    if len(all_possible_actions) == 1:
                        main_prob = 1
                        additional_prob = 0
                    else:
                        main_prob = make_action_proba
                        additional_prob = (1 - make_action_proba) / (len(all_possible_actions) - 1)

                    add_value_sum = 0
                    for another_action in possible_actions:
                        if another_action != action:
                            another_coord = state + np.array(env.maze_view.maze.STEPS[another_action])
                            add_value_sum += v[tuple(another_coord)]

                    temp_values.append(value * main_prob + additional_prob * add_value_sum)

                v[state] = env.calculate_reward(state) + gamma * np.max(temp_values)
                policy[state] = possible_actions[np.argmax(temp_values)]

                best_chance = max(best_chance, np.abs(old_value - v[state]))

        if best_chance < epsilon:
            break
        iteration += 1

        try:
            state = env.reset()
        except:
            state = np.zeros(2)

        for t in range(50):
            # Select an action
            action = policy[tuple(state)]
            # execute the action
            if np.random.choice([True, False], p=[make_action_proba, 1 - make_action_proba]):
                state, reward, done, _ = env.step(action)
                reward = v[tuple(state)]
            else:
                action = env.action_space.sample()
                state, reward, done, _ = env.step(action)
                reward = v[tuple(state)]
            # Observe the result
            total_reward += reward

            tmp_steps.append(t)

            if debug:
                if done or t >= max_t - 1:
                    print('Iteration: {}'.format(iteration))
                    print("t = %d" % t)
                    print("Streaks: %d" % num_streaks)
                    print("Total reward: %f" % total_reward)
                    print("Average Streaks: %f" % (num_streaks / (t + 1)))
                    print("Average Rewards: %f" % (total_reward / (t + 1)))
                    print("")

            if render_maze:
                env.render()

            if env.is_game_over():
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
            if num_streaks > max_win_streak:
                print('BREAK')
                break

            total_rewards.append(total_reward)
            total_streaks.append(num_streaks)
            steps_qty.append(tmp_steps[-1])

    return total_rewards, total_streaks, steps_qty


def value_iteration_main(env, gamma, epsilon, make_action_proba, max_win_streak=100, debug=True,
                         render_maze=True, recording_folder='./videos', enable_recording=True
                         ):

    env = Monitor(env, recording_folder, video_callable=lambda episode: True, force=True)

    all_states = [(x, y) for x in range(env.maze_size[0]) for y in range(env.maze_size[0])]
    policy = {}
    v = {}

    for s in all_states:
        policy[s] = np.random.choice(env.get_all_possible_actions(s))

    all_possible_actions = {state: env.get_all_possible_actions(state) for state in all_states}

    for s in all_states:
        reward = env.calculate_reward(s)
        v[s] = reward

    if enable_recording:
        env._start(recording_folder, video_callable=lambda episode: True, force=True)

    total_rewards, total_streaks, steps_qty = simulate_value_iteration(env=env, v=v, policy=policy, gamma=gamma,
                                                                       epsilon=epsilon,
                                                                       max_win_streak=max_win_streak,
                                                                       make_action_proba=make_action_proba, debug=debug,
                                                                       render_maze=render_maze,
                                                                       all_possible_actions=all_possible_actions)

    return total_rewards, total_streaks, steps_qty

