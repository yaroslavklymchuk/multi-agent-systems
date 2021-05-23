import re
from PIL import Image, ImageDraw


class BadChessboard(ValueError):
    pass


def expand_blanks(fen):
    '''Expand the digits in an FEN string into spaces

    >>> expand_blanks("rk4q3")
    'rk    q   '
    '''
    def expand(match):
        return ' ' * int(match.group(0))
    return re.compile(r'\d').sub(expand, fen)


def check_valid(expanded_fen, n):
    '''Asserts an expanded FEN string is valid'''
    print(expanded_fen)
    match = re.compile(r'([KQBNRPkqbnrp ]{{{n}}}/){{{n}}}$'.format(n=n)).match
    if not match(expanded_fen + '/'):
        raise BadChessboard()


def expand_fen(fen, n):
    '''Preprocesses a fen string into an internal format.

    Each square on the chessboard is represented by a single
    character in the output string. The rank separator characters
    are removed. Invalid inputs raise a BadChessboard error.
    '''
    expanded = expand_blanks(fen)
    check_valid(expanded, n)
    return expanded.replace('/', '')


def draw_board(n=8, sq_size=(20, 20)):
    '''Return an image of a chessboard.

    The board has n x n squares each of the supplied size.'''
    from itertools import cycle
    def square(i, j):
        return i * sq_size[0], j * sq_size[1]
    opaque_grey_background = 192, 255
    board = Image.new('LA', square(n, n), opaque_grey_background)
    draw_square = ImageDraw.Draw(board).rectangle
    whites = ((square(i, j), square(i + 1, j + 1))
              for i_start, j in zip(cycle((0, 1)), range(n))
              for i in range(i_start, n, 2))
    for white_square in whites:
        draw_square(white_square, fill='white')#'white')
    return board


class DrawChessPosition(object):
    '''Chess position renderer.

    Create an instance of this class, then call
    '''
    def __init__(self, n):
        '''Initialise, preloading pieces and creating a blank board.'''
        self.n = n
        self.create_pieces()
        self.create_blank_board()

    def create_pieces(self):
        '''Load the chess pieces from disk.

        Also extracts and caches the alpha masks for these pieces.
        '''
        whites = 'KQBNRP'
        piece_images = dict(
            zip(whites, (Image.open('pieces/%s.png' % p) for p in whites)))
        blacks = 'kqbnrp'
        piece_images.update(dict(
            zip(blacks, (Image.open('pieces/%s.png' % p) for p in blacks))))
        piece_sizes = set(piece.size for piece in piece_images.values())
        # Sanity check: the pieces should all be the same size
        assert len(piece_sizes) == 1
        self.piece_w, self.piece_h = piece_sizes.pop()
        self.piece_images = piece_images
        self.piece_masks = dict((pc, img.split()[3]) for pc, img in
                                 self.piece_images.items())

    def create_blank_board(self):
        '''Pre-render a blank board.'''
        self.board = draw_board(self.n, sq_size=(self.piece_w, self.piece_h))

    def point(self, i, j):
        '''Return the top left of the square at (i, j).'''
        w, h = self.piece_w, self.piece_h
        return i * h, j * w

    def square(self, i, j):
        '''Return the square at (i, j).'''
        t, l = self.point(i, j)
        b, r = self.point(i + 1, j + 1)
        return t, l, b, r

    def draw(self, fen):
        '''Return an image depicting the input position.

        fen - the first record of a FEN chess position.
        Clients are responsible for resizing this image and saving it,
        if required.
        '''
        board = self.board.copy()
        pieces = expand_fen(fen, self.n)
        images, masks, n = self.piece_images, self.piece_masks, self.n
        pts = (self.point(i, j) for j in range(n) for i in range(n))
        def not_blank(pt_pc):
            return pt_pc[1] != ' '
        for pt, piece in filter(not_blank, zip(pts, pieces)):
            board.paste(images[piece], pt, masks[piece])
        return board
