import pygame
import numpy as np

from maze import Maze
from copy import deepcopy


class MazeView2D:

    def __init__(self, maze_name="Maze2D",
                 maze_size=(30, 30), screen_size=(600, 600),
                 enable_render=True):

        # PyGame configurations
        pygame.init()
        pygame.display.set_caption(maze_name)
        self.clock = pygame.time.Clock()
        self.__game_over = False
        self.__enable_render = enable_render

        self.__maze = Maze(maze_size=maze_size)

        self.maze_size = self.__maze.maze_size
        if self.__enable_render is True:
            # to show the right and bottom border
            self.screen = pygame.display.set_mode(screen_size)
            self.__screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))

        # Set the starting point
        self.__entrance = np.zeros(2, dtype=int)

        # Set the Goal
        self.__goal = np.array(self.maze_size) - np.array((1, 1))

        # Set the Robot (box)
        self.__robot = self.entrance + np.array((1, 0))

        # Load a background
        background = pygame.image.load('ocean.jpg')
        background = pygame.transform.scale(background, (700, 700))

        # Load a entrance
        self.entr = pygame.image.load('entrance.png')
        self.entr = pygame.transform.scale(self.entr, (64, 64))

        # Load a exit
        self.exit = pygame.image.load('end.png')
        self.exit = pygame.transform.scale(self.exit, (64, 64))

        # Load a player
        self.player_box = pygame.image.load('wooden_box.png')
        self.player_box = pygame.transform.scale(self.player_box, (45, 45))

        # Load a turtle
        self.player_turtle = pygame.image.load('turtle.png')
        # self.box = pygame.transform.scale(self.box, (45, 45))
        self.player2 = self.entrance

        # Caption and icon
        pygame.display.set_caption("Ocean Life")
        icon = pygame.image.load('turtle.png')
        pygame.display.set_icon(icon)

        if self.__enable_render is True:
            # Create a background
            self.background = pygame.Surface(self.screen.get_size()).convert()
            # self.background.fill((255, 255, 255))

            # background image
            self.background.blit(background, (0, 0))

            # Create a layer for the maze
            self.maze_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
            self.maze_layer.fill((0, 0, 0, 0,))

            # show the maze
            self.__draw_maze()

            # show the portals
            # self.__draw_portals()

            # show the robot (box)
            self.__draw_robot()

            # show the turtle
            self.__draw_turtle()

            # show the entrance
            self.__draw_entrance()

            # show the goal
            self.__draw_goal()

    def update(self, mode="human"):
        try:
            img_output = self.__view_update(mode)
            self.__controller_update()
        except Exception as e:
            self.__game_over = True
            self.quit_game()
            raise e
        else:
            return img_output

    def quit_game(self):
        try:
            self.__game_over = True
            if self.__enable_render is True:
                pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    def move_turtle(self, dir):
        if dir not in self.__maze.STEPS.keys():
            raise ValueError("dir cannot be %s. The only valid dirs are %s."
                             % (str(dir), str(self.__maze.STEPS.keys())))

        if self.__maze.is_open(self.player2, dir):

            # update the drawing
            self.__draw_turtle(transparency=0)
            # if it's in a portal afterward
            self.player2 = self.previous
            self.__draw_turtle(transparency=255)
    # move box

    def move_robot(self, dir):
        if dir not in self.__maze.STEPS.keys():
            raise ValueError("dir cannot be %s. The only valid dirs are %s."
                             % (str(dir), str(self.__maze.STEPS.keys())))

        if self.__maze.is_open(self.__robot, dir):

            # update the drawing
            self.__draw_robot(transparency=0)

            # move the robot
            self.previous = deepcopy(self.__robot)
            self.__robot += np.array(self.__maze.STEPS[dir])

    def reset_robot(self):
        self.__draw_robot(transparency=0)
        self.__robot = np.array((1, 0))

    def reset_turtle(self):
        # self.__draw_robot()
        self.__draw_turtle(transparency=0)
        self.player2 = np.zeros(2, dtype=int)

    def __controller_update(self):
        if not self.__game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.__game_over = True
                    self.quit_game()

    def __view_update(self, mode="human"):
        if not self.__game_over:
            # update the robot and box's position
            self.__draw_entrance()
            self.__draw_goal()
            self.__draw_turtle()
            self.__draw_robot()

            # update the screen
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.maze_layer, (0, 0))

            if mode == "human":
                pygame.display.flip()

            return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

    def __draw_maze(self):
        if self.__enable_render is False:
            return

        line_colour = (255, 255, 255, 255)

        # drawing the horizontal lines
        for y in range(self.maze.MAZE_H + 1):
            pygame.draw.line(self.maze_layer, line_colour, (0, y * self.CELL_H),
                             (self.SCREEN_W, y * self.CELL_H))

        # drawing the vertical lines
        for x in range(self.maze.MAZE_W + 1):
            pygame.draw.line(self.maze_layer, line_colour, (x * self.CELL_W, 0),
                             (x * self.CELL_W, self.SCREEN_H))

        # breaking the walls
        for x in range(len(self.maze.maze_cells)):
            for y in range(len(self.maze.maze_cells[x])):
                # check the which walls are open in each cell
                walls_status = self.maze.get_walls_status(self.maze.maze_cells[x, y])
                dirs = ""
                for dir, open in walls_status.items():
                    if open:
                        dirs += dir
                self.__cover_walls(x, y, dirs)

    def __cover_walls(self, x, y, dirs, colour=(0, 0, 255, 15)):
        if self.__enable_render is False:
            return

        dx = x * self.CELL_W
        dy = y * self.CELL_H

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")

        for dir in dirs:
            if dir == "S":
                line_head = (dx + 1, dy + self.CELL_H)
                line_tail = (dx + self.CELL_W - 1, dy + self.CELL_H)
            elif dir == "N":
                line_head = (dx + 1, dy)
                line_tail = (dx + self.CELL_W - 1, dy)
            elif dir == "W":
                line_head = (dx, dy + 1)
                line_tail = (dx, dy + self.CELL_H - 1)
            elif dir == "E":
                line_head = (dx + self.CELL_W, dy + 1)
                line_tail = (dx + self.CELL_W, dy + self.CELL_H - 1)
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            pygame.draw.line(self.maze_layer, colour, line_head, line_tail)

    def __draw_robot(self, transparency=0):
        if self.__enable_render is False:
            return

        x = int(self.__robot[0] * self.CELL_W + self.CELL_W * 0.01 + 0.01)
        y = int(self.__robot[1] * self.CELL_H + self.CELL_H * 0.01 + 0.01)

        self.screen.blit(self.player_box, (x, y))
        pygame.display.flip()

    def __draw_turtle(self, transparency=0):
        if self.__enable_render is False:
            return

        x = int(self.player2[0] * self.CELL_W + self.CELL_W * 0.01 + 0.01)
        y = int(self.player2[1] * self.CELL_H + self.CELL_H * 0.01 + 0.01)

        self.screen.blit(self.player_turtle, (x, y))
        pygame.display.flip()

    def __draw_entrance(self):
        x_entr = int(self.entrance[0])
        y_entr = int(self.entrance[1])
        self.background.blit(self.entr, (x_entr, y_entr))

    def __draw_goal(self):
        x_exit = int(self.goal[0] * self.CELL_W + 0.5 + 1)
        y_exit = int(self.goal[1] * self.CELL_H + 0.5 + 1)
        self.background.blit(self.exit, (x_exit, y_exit))

    def __colour_cell(self, cell, colour, transparency):

        if self.__enable_render is False:
            return

        if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
            raise TypeError("cell must a be a tuple, list, or numpy array of size 2")

        x = int(cell[0] * self.CELL_W + 0.5 + 1)
        y = int(cell[1] * self.CELL_H + 0.5 + 1)
        w = int(self.CELL_W + 0.5 - 1)
        h = int(self.CELL_H + 0.5 - 1)
        pygame.draw.rect(self.maze_layer, colour + (transparency,), (x, y, w, h))

    @property
    def maze(self):
        return self.__maze

    @property
    def robot(self):
        return self.__robot

    @property
    def entrance(self):
        return self.__entrance

    @property
    def goal(self):
        return self.__goal

    @property
    def game_over(self):
        return self.__game_over

    @property
    def SCREEN_SIZE(self):
        return tuple(self.__screen_size)

    @property
    def SCREEN_W(self):
        return int(self.SCREEN_SIZE[0])

    @property
    def SCREEN_H(self):
        return int(self.SCREEN_SIZE[1])

    @property
    def CELL_W(self):
        return float(self.SCREEN_W) / float(self.maze.MAZE_W)

    @property
    def CELL_H(self):
        return float(self.SCREEN_H) / float(self.maze.MAZE_H)

