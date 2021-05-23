from itertools import product
from math import hypot, sqrt

MAX_DESK_SIZE = 100


class Figure:
    def __init__(self, name, coordinates):
        self._name = name
        self._coordinates = coordinates

    def get_step_coordinates(self, board_size):
        pass

    def attack(self, target_coordinates, board_size):
        list_possible_coordinates = self.get_step_coordinates(board_size)
        return target_coordinates in list_possible_coordinates

    def is_possible_attack(self, desk, desk_size):
        list_possible_coordinates = self.get_step_coordinates(desk_size)
        return any((desk[el[0], el[1]] for el in list_possible_coordinates
                    if (0 <= el[0] < desk_size) and (0 <= el[1] < desk_size))
                   )


class Pawn(Figure):
    def __init__(self, name, coordinates):
        super().__init__(name, coordinates)

    def get_step_coordinates(self, board_size):
        x, y = self._coordinates

        all_coordinates = list(product(range(board_size), range(board_size)))
        possible_coordinates = (coord for coord in all_coordinates
                                if (abs(x - coord[0]) == 1 and y - coord[1] == 1)
                                )

        possible_coordinates = list(set(possible_coordinates))

        return possible_coordinates


class Knight(Figure):
    def __init__(self, name, coordinates):
        super().__init__(name, coordinates)

    def get_step_coordinates(self, board_size):
        x, y = self._coordinates
        all_coordinates = list(product(range(board_size), range(board_size)))

        possible_coordinates = (coord for coord in all_coordinates
                                if ((abs(coord[0] - x) == 2 and abs(coord[1] - y) == 1)
                                    or (abs(coord[0] - x) == 1 and abs(coord[1] - y) == 2))
                                and coord != (x, y)
                                )
        possible_coordinates = list(set(possible_coordinates))

        return possible_coordinates


class Rook(Figure):
    def __init__(self, name, coordinates):
        super().__init__(name, coordinates)

    def get_step_coordinates(self, board_size):
        x, y = self._coordinates

        all_coordinates = list(product(range(board_size), range(board_size)))

        possible_coordinates = (coord for coord in all_coordinates
                                if (coord[0] == x or coord[1] == y) and coord != (x, y)
                                )

        possible_coordinates = list(set(possible_coordinates))

        return possible_coordinates


class Bishop(Figure):
    def __init__(self, name, coordinates):
        super().__init__(name, coordinates)

    def get_step_coordinates(self, board_size):
        x, y = self._coordinates

        all_coordinates = list(product(range(board_size), range(board_size)))

        possible_coordinates = (coord for coord in all_coordinates
                                if abs(coord[0] - x) == abs(coord[1] - y) and coord != (x, y)
                                )

        possible_coordinates = list(set(possible_coordinates))

        return possible_coordinates


class Queen(Figure):
    def __init__(self, name, coordinates):
        super().__init__(name, coordinates)

    def get_step_coordinates(self, board_size):
        x, y = self._coordinates

        bishop = Bishop(self._name, self._coordinates)
        rook = Rook(self._name, self._coordinates)

        rook_possible_coords = rook.get_step_coordinates(board_size)
        bishop_possible_coords = bishop.get_step_coordinates(board_size)

        list_possible_coordinates = rook_possible_coords + bishop_possible_coords
        list_possible_coordinates = list(set(list_possible_coordinates))
        list_possible_coordinates = [coord for coord in list_possible_coordinates if coord != (x, y)]

        return list_possible_coordinates


class King(Figure):
    def __init__(self, name, coordinates):
        super().__init__(name, coordinates)

    def get_step_coordinates(self, board_size):
        x, y = self._coordinates

        all_coordinates = list(product(range(board_size), range(board_size)))

        possible_coordinates = (coord for coord in all_coordinates
                                if hypot(x - coord[0], y - coord[1]) <= sqrt(2) and coord != (x, y)
                                )

        possible_coordinates = list(set(possible_coordinates))

        return possible_coordinates
