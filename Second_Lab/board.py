import rstr
from collections import Counter
import numpy as np

from visualize import DrawChessPosition
from figures import King, Queen, Pawn, Rook, Knight, Bishop

REGXC = r'([{figures} ]{{{n}}}/){{{n}}}$'
CLASSES_MAPPING = {'B': Bishop, 'N': Knight, 'P': Pawn, 'R': Rook, 'K': King, 'Q': Queen}


class ChessBoard:

    def __init__(
            self,
            size,
            config,
            regex=REGXC
    ):
        self.size = size
        self.config = config
        self.regex = regex

    def generate_board(self):
        regex_tmp = self.regex.format(figures=('').join(self.config.keys()), n=self.size)
        gen = rstr.xeger(regex_tmp)
        gen = ('/').join(gen.split('/')[:-1])

        counter = Counter(gen)
        for piece, qty in self.config.items():
            real_qty = counter.get(piece)
            while real_qty > qty:
                for i, el in enumerate(gen):
                    if el == piece and np.random.uniform(0, 1) > 0.5 and real_qty > qty:
                        gen = gen[:i] + ' ' + gen[i + 1:]
                        real_qty -= 1

        renderer = DrawChessPosition(n=self.size)
        board = renderer.draw(gen)
        return board, gen

    def get_coordinates_map(self, board, gen):
        coords = np.zeros((self.size, self.size))
        coords_words = np.zeros((self.size, self.size)).astype(str)
        coords_dict = {}

        for col in self.config.keys():
            coords_dict[col] = []

        for i, row in enumerate(gen.split('/')):
            for j, col in enumerate(row):
                number = self.config.get(col)
                if number:
                    coords[i][j] = number
                    coords_words[i][j] = col
                    coords_dict[col].append((i, j))

        coords_words = [[' ' if el1 == '0.0' else el1 for el1 in el] for el in coords_words]

        figures_mapping = {}

        for key, coords_figure in coords_dict.items():
            for coord in coords_figure:
                figures_mapping[coord] = CLASSES_MAPPING.get(key)(key, coord)

        return coords, coords_words, coords_dict, figures_mapping

    def show_field(self):
        board, gen = self.generate_board()
        board.show()
