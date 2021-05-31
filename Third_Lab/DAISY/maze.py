import os
import numpy as np
import random


class Maze:
    STEPS = {
        "N": (0, -1),
        "E": (1, 0),
        "S": (0, 1),
        "W": (-1, 0)
    }

    def __init__(self, maze_size=(10, 10)):

        self.maze_size = maze_size
        self._generate_maze()

    def save_maze(self, file_path):

        if not isinstance(file_path, str):
            raise TypeError("Invalid file_path. It must be a str.")

        if not os.path.exists(os.path.dirname(file_path)):
            raise ValueError("Cannot find the directory for %s." % file_path)

        else:
            np.save(file_path, self.maze_cells, allow_pickle=False, fix_imports=True)

    @classmethod
    def load_maze(cls, file_path):

        if not isinstance(file_path, str):
            raise TypeError("Invalid file_path. It must be a str.")

        if not os.path.exists(file_path):
            raise ValueError("Cannot find %s." % file_path)

        else:
            return np.load(file_path, allow_pickle=False, fix_imports=True)

    def _generate_maze(self):
        self.maze_cells = np.zeros(self.maze_size, dtype=int)

        current_cell = (random.randint(0, self.MAZE_W - 1), random.randint(0, self.MAZE_H - 1))
        num_cells_visited = 1
        cell_stack = [current_cell]
        while cell_stack:

            current_cell = cell_stack.pop()
            x0, y0 = current_cell

            neighbours = dict()
            for dir_key, dir_val in self.STEPS.items():
                x1 = x0 + dir_val[0]
                y1 = y0 + dir_val[1]
                # 0 <= x1 < self.MAZE_W and 0 <= y1 < self.MAZE_H:
                if self.is_within_bound(x1, y1):
                    # if all four walls still exist
                    if self.all_walls_intact(self.maze_cells[x1, y1]):
                        neighbours[dir_key] = (x1, y1)

            if neighbours:
                dir = random.choice(tuple(neighbours.keys()))
                x1, y1 = neighbours[dir]
                self.maze_cells[x1, y1] = self.__break_walls(self.maze_cells[x1, y1], self.__get_opposite_wall(dir))

                cell_stack.append(current_cell)
                cell_stack.append((x1, y1))
                num_cells_visited += 1

    def __break_random_walls(self, percent):
        # find some random cells to break
        num_cells = int(round(self.MAZE_H * self.MAZE_W * percent))
        cell_ids = random.sample(range(self.MAZE_W * self.MAZE_H), num_cells)

        # for each of those walls
        for cell_id in cell_ids:
            x = cell_id % self.MAZE_H
            y = int(cell_id / self.MAZE_H)

            # randomize the compass order
            dirs = random.sample(list(self.STEPS.keys()), len(self.STEPS))
            for dir in dirs:
                # break the wall if it's not already open
                if self.is_breakable((x, y), dir):
                    self.maze_cells[x, y] = self.__break_walls(self.maze_cells[x, y], dir)
                    break

    def is_open(self, cell_id, dir):
        # check if it would be out-of-bound
        x1 = cell_id[0] + self.STEPS[dir][0]
        y1 = cell_id[1] + self.STEPS[dir][1]

        # if cell is still within bounds after the move
        if self.is_within_bound(x1, y1):
            # check if the wall is opened
            this_wall = bool(self.get_walls_status(self.maze_cells[cell_id[0], cell_id[1]])[dir])
            other_wall = bool(self.get_walls_status(self.maze_cells[x1, y1])[self.__get_opposite_wall(dir)])
            return this_wall or other_wall
        return False

    def is_breakable(self, cell_id, dir):
        # check if it would be out-of-bound
        x1 = cell_id[0] + self.STEPS[dir][0]
        y1 = cell_id[1] + self.STEPS[dir][1]

        return not self.is_open(cell_id, dir) and self.is_within_bound(x1, y1)

    def is_within_bound(self, x, y):
        # true if cell is still within bounds after the move
        return 0 <= x < self.MAZE_W and 0 <= y < self.MAZE_H

    @property
    def MAZE_W(self):
        return int(self.maze_size[0])

    @property
    def MAZE_H(self):
        return int(self.maze_size[1])

    @classmethod
    def get_walls_status(cls, cell):
        walls = {
            "N": (cell & 0x1) >> 0,
            "E": (cell & 0x2) >> 1,
            "S": (cell & 0x4) >> 2,
            "W": (cell & 0x8) >> 3,
        }
        return walls

    @classmethod
    def all_walls_intact(cls, cell):
        return cell & 0xF == 0

    @classmethod
    def num_walls_broken(cls, cell):
        walls = cls.get_walls_status(cell)
        num_broken = 0
        for wall_broken in walls.values():
            num_broken += wall_broken
        return num_broken

    @classmethod
    def __break_walls(cls, cell, dirs):
        if "N" in dirs:
            cell |= 0x1
        if "E" in dirs:
            cell |= 0x2
        if "S" in dirs:
            cell |= 0x4
        if "W" in dirs:
            cell |= 0x8
        return cell

    @classmethod
    def __get_opposite_wall(cls, dirs):

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")

        opposite_dirs = ""

        for dir in dirs:
            if dir == "N":
                opposite_dir = "S"
            elif dir == "S":
                opposite_dir = "N"
            elif dir == "E":
                opposite_dir = "W"
            elif dir == "W":
                opposite_dir = "E"
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            opposite_dirs += opposite_dir

        return opposite_dirs
