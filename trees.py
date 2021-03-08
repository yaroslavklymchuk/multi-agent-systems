import math

import numpy as np
from mesa import Agent


class BaseTree(Agent):
    """Class that implements basic Tree instance.
    Tree has three main states: 0 - 'Green Tree', 1 - 'Firing Tree', 2 - 'Burned Tree'
    """
    def __init__(self, model, coordinates, spread_probability=1):
        super().__init__(coordinates, model)
        self.coordinates = coordinates
        self.unique_id = coordinates
        self.condition = 0
        self.spread_probability = spread_probability

    def step(self):
        if self.condition == 1:
            neighbors = self.model.grid.get_neighbors(self.coordinates, moore=False, include_center=False)
            for neighbor in neighbors:
                if neighbor.condition == 0 and self.random.random() < self.spread_probability:
                    neighbor.condition = 1
            self.condition = 2


class WindTree(Agent):
    """Class that implements Tree instance with wind extension.
    Tree has three main states: 0 - 'Green Tree', 1 - 'Firing Tree', 2 - 'Burned Tree'
    """
    def __init__(self, model, coordinates, wind_speed, spread_probability=1):
        super().__init__(coordinates, model)
        self.coordinates = coordinates
        self.unique_id = coordinates
        self.condition = 0
        self.spread_probability = spread_probability
        self.wind_speed = wind_speed
        self.wind_directions = [0, 90, 180, 270]

    def step(self):
        if self.condition == 1:
            neighbors = self.model.grid.get_neighbors(self.coordinates, moore=False, include_center=False)
            for neighbor in neighbors:
                if neighbor.condition == 0:
                    dx = neighbor.coordinates[0] - self.coordinates[0]
                    dy = neighbor.coordinates[1] - self.coordinates[1]
                    fire_direction = round(math.degrees(math.atan2(dy, dx))) + 45
                    print(fire_direction)
                    wind_direction = np.random.choice(self.wind_directions)
                    spread_proba = self.spread_probability

                    if fire_direction == wind_direction:
                        spread_proba += self.wind_speed
                    else:
                        spread_proba -= self.wind_speed
                    if self.random.random() < spread_proba:
                        neighbor.condition = 1
            self.condition = 2
