import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from view_maze import MazeView2D


class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_size=None, enable_render=True):

        self.viewer = None
        self.enable_render = enable_render

        self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                    maze_size=maze_size, screen_size=(640, 640),
                                    enable_render=enable_render)

        self.maze_size = self.maze_view.maze_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=int)
        high = np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self.configure()

    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def check_done(self, state):
        return np.array_equal(state, self.maze_view.goal)

    def calculate_reward(self, state):

        if np.array_equal(state, self.maze_view.goal):
            reward = 1
        else:
            reward = -0.1 / (self.maze_size[0]*self.maze_size[1])
            #reward = 0

        return reward

    def step(self, action):
        if isinstance(action, int):
            self.maze_view.move_robot(self.ACTION[action])
            self.maze_view.move_turtle(self.ACTION[action])

        else:
            self.maze_view.move_robot(action)
            self.maze_view.move_turtle(action)

        done = self.check_done(self.maze_view.robot)
        reward = self.calculate_reward(self.maze_view.robot)

        self.state = self.maze_view.robot

        info = {}

        return self.state, reward, done, info

    def get_all_possible_actions(self, state):

        possible_actions = [direction for direction in self.maze_view.maze.STEPS.keys()
                            if self.maze_view.maze.is_open(state, direction)
                            ]

        possible_actions = list(set(possible_actions))

        return possible_actions

    def reset(self):
        self.maze_view.reset_robot()
        self.state = np.zeros(2)
        self.steps_beyond_done = None
        self.done = False
        return self.state

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode="human", close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)
