import math
import pygame
import chess

WIDTH = 640
HEIGHT = 640
SQ_SIZE = WIDTH // 8

IMAGES = {}

# -------------------------------------
# Promotion Helpers
# -------------------------------------

def is_true_promotion_move(board, from_sq, to_sq):
    """Return True if ANY legal promotion (including capture) exists."""
    for promo_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        test_move = chess.Move(from_sq, to_sq, promotion=promo_piece)
        if test_move in board.legal_moves:
            return True
    return False


def draw_promotion_menu(screen, color, col, gui_row):
    menu_width = SQ_SIZE
    menu_height = 4 * SQ_SIZE

    if color == chess.WHITE:
        y = gui_row * SQ_SIZE
    else:
        y = gui_row * SQ_SIZE - (3 * SQ_SIZE)

    x = col * SQ_SIZE

    pygame.draw.rect(screen, (230, 230, 230), (x, y, menu_width, menu_height))

    pieces = ["q", "r", "b", "n"]

    boxes = []
    for i, p in enumerate(pieces):
        key = ("w" if color == chess.WHITE else "b") + p
        img = IMAGES[key]
        box_y = y + i * SQ_SIZE
        screen.blit(img, pygame.Rect(x, box_y, SQ_SIZE, SQ_SIZE))
        boxes.append((x, box_y, SQ_SIZE, SQ_SIZE, p))

    pygame.display.flip()
    return boxes


# -------------------------------------
# Load Images
# -------------------------------------

def load_images():
    pieces = ["wp", "wn", "wb", "wr", "wq", "wk",
              "bp", "bn", "bb", "br", "bq", "bk"]

    for p in pieces:
        IMAGES[p] = pygame.transform.scale(
            pygame.image.load(f"pieces/{p}.png"), (SQ_SIZE, SQ_SIZE)
        )


# -------------------------------------
# Drawing Functions
# -------------------------------------

def draw_board(screen):
    colors = [(240, 217, 181), (181, 136, 99)]
    for r in range(8):
        for c in range(8):
            pygame.draw.rect(
                screen,
                colors[(r + c) % 2],
                pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            )


def draw_pieces(screen, board):
    for square, piece in board.piece_map().items():
        col = chess.square_file(square)
        row = 7 - chess.square_rank(square)
        symbol = piece.symbol()
        img = IMAGES["w" + symbol.lower()] if symbol.isupper() else IMAGES["b" + symbol.lower()]
        screen.blit(img, pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))


def highlight_squares(screen, board, selected_square):
    if selected_square is None:
        return

    col = chess.square_file(selected_square)
    row = 7 - chess.square_rank(selected_square)

    s = pygame.Surface((SQ_SIZE, SQ_SIZE))
    s.set_alpha(120)
    s.fill((246, 246, 105))
    screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))

    for move in board.legal_moves:
        if move.from_square == selected_square:
            dc = chess.square_file(move.to_square)
            dr = 7 - chess.square_rank(move.to_square)
            center = (dc * SQ_SIZE + SQ_SIZE // 2, dr * SQ_SIZE + SQ_SIZE // 2)
            pygame.draw.circle(screen, (50, 50, 50), center, 12)


# -------------------------------------
# Evaluation
# -------------------------------------

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
EXTENDED_CENTER = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4,
    chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6
}


def evaluate_board(board):
    if board.is_checkmate():
        return -10000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    material_white = material_black = 0
    center_white = center_black = 0

    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        bonus = 20 if square in CENTER_SQUARES else 10 if square in EXTENDED_CENTER else 0

        if piece.color == chess.WHITE:
            material_white += value
            center_white += bonus
        else:
            material_black += value
            center_black += bonus

    score = (material_white - material_black) + (center_white - center_black)

    # Mobility
    my_moves = board.legal_moves.count()
    board.push(chess.Move.null())
    opp_moves = board.legal_moves.count()
    board.pop()
    score += 5 * (my_moves - opp_moves)

    # King safety
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)

    if wk in (chess.G1, chess.C1):
        score += 40
    if bk in (chess.G8, chess.C8):
        score -= 40

    if board.fullmove_number > 10:
        if wk in (chess.E1, chess.D1, chess.F1):
            score -= 30
        if bk in (chess.E8, chess.D8, chess.F8):
            score += 30

    return score if board.turn == chess.WHITE else -score


# -------------------------------------
# Engine
# -------------------------------------

def ordered_moves(board):
    moves = list(board.legal_moves)

    def mv_score(move):
        score = 0
        if board.is_capture(move):
            score += 10000
        if move.promotion:
            score += 5000
        if move.to_square in CENTER_SQUARES:
            score += 500
        return score

    moves.sort(key=mv_score, reverse=True)
    return moves


def negamax(board, depth, alpha, beta):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    maxv = -math.inf

    for move in ordered_moves(board):
        board.push(move)
        val = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if val > maxv:
            maxv = val
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break

    return maxv


class AIPlayer:
    def __init__(self, depth=15):
        self.depth = depth

    def choose_move(self, board):
        best = None
        best_score = -math.inf

        for move in ordered_moves(board):
            board.push(move)
            score = -negamax(board, self.depth - 1, -math.inf, math.inf)
            board.pop()

            if score > best_score:
                best_score = score
                best = move

        return best


# -------------------------------------
# Main Loop
# -------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Python Chess GUI")

    board = chess.Board()
    load_images()
    ai = AIPlayer(depth=3)

    selected = None
    promoting = False
    promo_from = None
    promo_to = None
    promo_boxes = []

    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            # Promotion click handler
            if promoting and event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                for (x, y, w, h, p) in promo_boxes:
                    if x <= mx <= x + w and y <= my <= y + h:
                        move = chess.Move(promo_from, promo_to,
                                          promotion={"q": chess.QUEEN,
                                                     "r": chess.ROOK,
                                                     "b": chess.BISHOP,
                                                     "n": chess.KNIGHT}[p])
                        if move in board.legal_moves:
                            board.push(move)
                        promoting = False
                        selected = None
                continue

            # Normal clicks
            if event.type == pygame.MOUSEBUTTONDOWN and not promoting:
                mx, my = event.pos
                row = my // SQ_SIZE
                col = mx // SQ_SIZE
                clicked = chess.square(col, 7 - row)

                if selected is None:
                    piece = board.piece_at(clicked)
                    if piece and piece.color == board.turn:
                        selected = clicked
                else:
                    # Check if PROPER legal promotion exists
                    if is_true_promotion_move(board, selected, clicked):
                        promoting = True
                        promo_from = selected
                        promo_to = clicked

                        promo_boxes = draw_promotion_menu(
                            screen, board.turn, col, row
                        )
                    else:
                        move = chess.Move(selected, clicked)
                        if move in board.legal_moves:
                            board.push(move)
                        selected = None

        # AI move
        if not promoting and not board.is_game_over() and board.turn == chess.BLACK:
            mv = ai.choose_move(board)
            if mv:
                board.push(mv)

        if not promoting:
            draw_board(screen)
            highlight_squares(screen, board, selected)
            draw_pieces(screen, board)
            pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
