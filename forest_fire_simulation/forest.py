from itertools import product

from mesa import Model
from mesa.time import RandomActivation
from mesa.space import Grid
from mesa.datacollection import DataCollector

from trees import BaseTree, WindTree


class BaseForestFire(Model):
    def __init__(self, height, width, density, spread_probability=1):
        self.height = height
        self.width = width
        self.density = density
        self.spread_probability = spread_probability
        self.schedule = RandomActivation(self)
        self.grid = Grid(height, width, torus=False)
        self.dc = DataCollector({"green_trees": lambda m: self.count_type(m, 0),
                                 "fire_trees": lambda m: self.count_type(m, 1),
                                 "burned_trees": lambda m: self.count_type(m, 2)
                                 })

        for x, y in product(range(self.width), range(self.height)):
            if self.random.random() < density:
                new_tree = BaseTree(self, (x, y), spread_probability)
                if y == 0:
                    new_tree.condition = 1
                self.grid[y][x] = new_tree
                self.schedule.add(new_tree)
        self.running = True

    def step(self):
        self.schedule.step()
        self.dc.collect(self)
        if self.count_type(self, 1) == 0:
            self.running = False

    @staticmethod
    def count_type(model, tree_condition):
        return len(list(filter(lambda tree: tree.condition == tree_condition, model.schedule.agents)))


class WindForestFire(Model):
    def __init__(self, height, width, density, wind_speed, spread_probability=1):
        self.height = height
        self.width = width
        self.density = density
        self.spread_probability = spread_probability
        self.wind_speed = wind_speed
        self.schedule = RandomActivation(self)
        self.grid = Grid(height, width, torus=False)
        self.dc = DataCollector({"green_trees": lambda m: self.count_type(m, 0),
                                 "fire_trees": lambda m: self.count_type(m, 1),
                                 "burned_trees": lambda m: self.count_type(m, 2)
                                 })

        for x, y in product(range(self.width), range(self.height)):
            if self.random.random() < density:
                new_tree = WindTree(self, (x, y), self.wind_speed, self.spread_probability)
                if y == 0:
                    new_tree.condition = 1
                self.grid[y][x] = new_tree
                self.schedule.add(new_tree)
        self.running = True

    def step(self):
        self.schedule.step()
        self.dc.collect(self)
        if self.count_type(self, 1) == 0:
            self.running = False

    @staticmethod
    def count_type(model, tree_condition):
        return len(list(filter(lambda tree: tree.condition == tree_condition, model.schedule.agents)))
