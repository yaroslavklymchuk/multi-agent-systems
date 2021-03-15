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

    def calculate_angle(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        angle = round(math.degrees(math.atan2(dy, dx)))
        angle = np.argmin(np.abs(np.array(self.wind_directions) - np.array([angle])))
        angle = self.wind_directions[angle]
        return angle

    def step(self):
        if self.condition == 1:
            neighbors = self.model.grid.get_neighbors(self.coordinates, moore=False, include_center=False)
            for neighbor in neighbors:
                if neighbor.condition == 0:

                    fire_direction = self.calculate_angle(self.coordinates, neighbor.coordinates)
                    wind_direction = np.random.choice(self.wind_directions)
                    spread_proba = self.spread_probability

                    if fire_direction == wind_direction:
                        spread_proba += self.wind_speed
                    else:
                        spread_proba -= self.wind_speed
                    if self.random.random() < spread_proba:
                        neighbor.condition = 1
            self.condition = 2
