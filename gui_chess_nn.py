import math
import pygame
import chess
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from functools import lru_cache

# ============================================
# Global Settings
# ============================================

WIDTH = 640
HEIGHT = 640
SQ_SIZE = WIDTH // 8

IMAGES = {}

# ============================================
# TensorFlow: Configure Once At Startup
# ============================================

# Optional: force TF to ignore GPU to avoid slow warmups / Metal logs
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    # If no GPU or config not supported, just ignore
    pass


# ============================================
# Load Neural Network
# ============================================

try:
    print("Trying to load model ...")
    NN_MODEL = load_model("eval_net.h5", compile=False)
    print("Neural network loaded successfully.")
    USE_NN = True
except Exception as e:
    print("Model load failed!")
    print("ERROR:", e)
    NN_MODEL = None
    USE_NN = False


# ============================================
# Board → Tensor Conversion (from FEN)
# ============================================

def board_to_tensor_fen(fen: str) -> np.ndarray:
    """
    Converts a FEN string into a (8, 8, 12) tensor.
    Planes:
        0..5  = white pawn, knight, bishop, rook, queen, king
        6..11 = black pawn, knight, bishop, rook, queen, king
    """
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 12), dtype=np.float32)

    # python-chess maps: square index -> Piece
    for square, piece in board.piece_map().items():
        # 0 at a1, 7 at h1, 56 at a8, 63 at h8
        rank = chess.square_rank(square)   # 0 (rank 1) .. 7 (rank 8)
        file = chess.square_file(square)   # 0 (file a) .. 7 (file h)

        # Our tensor index: [row, col] with row 0 at top (rank 8)
        row = 7 - rank
        col = file

        channel = piece.piece_type - 1     # 0..5
        if piece.color == chess.BLACK:
            channel += 6

        tensor[row, col, channel] = 1.0

    return tensor


# ============================================
# Cached Neural Net Evaluation
# ============================================

@lru_cache(maxsize=50000)
def nn_eval_board(fen: str) -> float:
    """
    Neural network evaluation cached by FEN.
    Returns a scalar in [-1, 1] from White's perspective.
    """
    if not USE_NN or NN_MODEL is None:
        return 0.0

    tensor = board_to_tensor_fen(fen)
    # Shape (1, 8, 8, 12)
    tensor = np.expand_dims(tensor, axis=0)

    # Predict without logging
    v = NN_MODEL.predict(tensor, verbose=0)
    return float(v[0][0])


# ============================================
# Promotion Helpers
# ============================================

def is_true_promotion_move(board, frm, to):
    for promo_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        if chess.Move(frm, to, promotion=promo_piece) in board.legal_moves:
            return True
    return False


def draw_promotion_menu(screen, color, col, gui_row):
    menu_height = 4 * SQ_SIZE
    y = gui_row * SQ_SIZE if color == chess.WHITE else gui_row * SQ_SIZE - 3 * SQ_SIZE
    x = col * SQ_SIZE

    pygame.draw.rect(screen, (230, 230, 230), (x, y, SQ_SIZE, menu_height))

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


# ============================================
# Load Piece Images
# ============================================

def load_images():
    pieces = ["wp", "wn", "wb", "wr", "wq", "wk",
              "bp", "bn", "bb", "br", "bq", "bk"]

    for p in pieces:
        IMAGES[p] = pygame.transform.scale(
            pygame.image.load(f"pieces/{p}.png"), (SQ_SIZE, SQ_SIZE)
        )


# ============================================
# GUI Drawing
# ============================================

def draw_board(screen):
    light = (240, 217, 181)
    dark = (181, 136, 99)

    for r in range(8):
        for c in range(8):
            color = light if (r + c) % 2 == 0 else dark
            pygame.draw.rect(
                screen,
                color,
                pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            )


def draw_pieces(screen, board):
    for sq, piece in board.piece_map().items():
        col = chess.square_file(sq)
        row = 7 - chess.square_rank(sq)
        symbol = piece.symbol()

        img = IMAGES["w" + symbol.lower()] if symbol.isupper() else IMAGES["b" + symbol.lower()]
        screen.blit(img, pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))


def highlight_squares(screen, board, selected):
    if selected is None:
        return

    col = chess.square_file(selected)
    row = 7 - chess.square_rank(selected)

    s = pygame.Surface((SQ_SIZE, SQ_SIZE))
    s.set_alpha(120)
    s.fill((246, 246, 105))
    screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))

    for move in board.legal_moves:
        if move.from_square == selected:
            dc = chess.square_file(move.to_square)
            dr = 7 - chess.square_rank(move.to_square)
            center = (dc * SQ_SIZE + SQ_SIZE // 2, dr * SQ_SIZE + SQ_SIZE // 2)
            pygame.draw.circle(screen, (50, 50, 50), center, 12)


# ============================================
# Static Evaluation (Side-to-move Perspective)
# ============================================

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}
EXT_CENTER = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4,
    chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6
}


def static_eval(board: chess.Board) -> int:
    """
    Returns score in centipawns from the side-to-move perspective.
    Positive means good for the side to move.
    """
    if board.is_checkmate():
        return -10000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    w_mat = b_mat = 0
    w_center = b_center = 0

    for sq, pc in board.piece_map().items():
        v = PIECE_VALUES[pc.piece_type]
        bonus = 20 if sq in CENTER else 10 if sq in EXT_CENTER else 0

        if pc.color == chess.WHITE:
            w_mat += v
            w_center += bonus
        else:
            b_mat += v
            b_center += bonus

    score = (w_mat - b_mat) + (w_center - b_center)

    # Mobility
    my_moves = board.legal_moves.count()
    board.push(chess.Move.null())
    opp_moves = board.legal_moves.count()
    board.pop()

    score += 5 * (my_moves - opp_moves)

    # Castling bonus
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

    # Side-to-move perspective
    return score if board.turn == chess.WHITE else -score


# ============================================
# Combined Evaluation (Static + NN)
# ============================================

def evaluate(board: chess.Board) -> int:
    """
    Final evaluation used by negamax.
    Returns score from side-to-move perspective in centipawns.
    """
    base = static_eval(board)

    if not USE_NN:
        return base

    fen = board.fen()
    nn_value = nn_eval_board(fen)        # [-1, 1] from White perspective
    nn_cp = nn_value * 2000.0            # scale to centipawns

    # Convert to side-to-move perspective
    if board.turn == chess.WHITE:
        nn_side = nn_cp
    else:
        nn_side = -nn_cp

    blended = 0.7 * nn_side + 0.3 * base
    return int(blended)


# ============================================
# Move Ordering
# ============================================

def ordered_moves(board: chess.Board):
    moves = list(board.legal_moves)

    def score_move(m: chess.Move) -> int:
        val = 0
        if board.is_capture(m):
            val += 10000
        if m.promotion:
            val += 5000
        if m.to_square in CENTER:
            val += 500
        return val

    moves.sort(key=score_move, reverse=True)
    return moves


# ============================================
# Negamax Search
# ============================================

def negamax(board: chess.Board, depth: int, alpha: float, beta: float) -> int:
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    best = -math.inf

    for move in ordered_moves(board):
        board.push(move)
        val = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if val > best:
            best = val
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break

    return best


class AIPlayer:
    def __init__(self, depth: int = 2):
        self.depth = depth

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        best_move = None
        best_score = -math.inf

        for move in ordered_moves(board):
            board.push(move)
            val = -negamax(board, self.depth - 1, -math.inf, math.inf)
            board.pop()

            if val > best_score:
                best_score = val
                best_move = move

        return best_move


# ============================================
# Main Loop
# ============================================

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess AI + Neural Net")

    board = chess.Board()
    load_images()

    # Start with depth 2 while testing NN. You can push back to 3 later.
    ai = AIPlayer(depth=2)

    selected = None
    promoting = False
    promo_from = promo_to = None
    promo_boxes = []

    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Promotion click
            if promoting and event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                for (x, y, w, h, p) in promo_boxes:
                    if x <= mx <= x + w and y <= my <= y + h:
                        move = chess.Move(
                            promo_from,
                            promo_to,
                            promotion={
                                "q": chess.QUEEN,
                                "r": chess.ROOK,
                                "b": chess.BISHOP,
                                "n": chess.KNIGHT,
                            }[p],
                        )
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
                    if is_true_promotion_move(board, selected, clicked):
                        promoting = True
                        promo_from = selected
                        promo_to = clicked
                        promo_boxes = draw_promotion_menu(screen, board.turn, col, row)
                    else:
                        move = chess.Move(selected, clicked)
                        if move in board.legal_moves:
                            board.push(move)
                        selected = None

        # AI move
        if not promoting and board.turn == chess.BLACK and not board.is_game_over():
            mv = ai.choose_move(board)
            if mv:
                board.push(mv)

        # Draw
        if not promoting:
            draw_board(screen)
            highlight_squares(screen, board, selected)
            draw_pieces(screen, board)
            pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
