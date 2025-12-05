import math
import pygame
import chess
import chess.engine
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from functools import lru_cache

# ============================================
# Engine Metrics
# ============================================

NODES_SEARCHED = 0
NODES_PRUNED = 0
NN_EVAL_COUNT = 0

# Current Stockfish evaluation (centipawns, White POV)
CURRENT_EVAL_CP = 0.0

# ============================================
# Global Settings
# ============================================

BOARD_SIZE = 640
EVAL_PANEL_WIDTH = 80  # width for eval bar on the right

WIDTH = BOARD_SIZE + EVAL_PANEL_WIDTH
HEIGHT = BOARD_SIZE
SQ_SIZE = BOARD_SIZE // 8

IMAGES = {}

# Quiescence depth limit to avoid explosion
QUIESCENCE_DEPTH = 4

# Font (initialized in main)
FONT = None

# ============================================
# TensorFlow: Configure Once At Startup
# ============================================

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    # Ignore if not applicable
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
# Stockfish Setup (for eval bar)
# ============================================

# Adjust this path to your Stockfish binary
STOCKFISH_PATH = "/usr/local/bin/stockfish"

try:
    SF_ENGINE = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    print("Stockfish loaded for eval bar.")
except Exception as e:
    SF_ENGINE = None
    print("WARNING: Could not start Stockfish. Eval bar disabled.")
    print("ERROR:", e)


def evaluate_with_stockfish(board: chess.Board) -> float:
    """
    Use Stockfish to evaluate the current position.
    Returns centipawns from White's perspective.
    """
    if SF_ENGINE is None:
        return 0.0

    try:
        # You can tune depth or switch to time-based limits
        info = SF_ENGINE.analyse(board, chess.engine.Limit(depth=12))
        score = info["score"].pov(chess.WHITE)

        if score.is_mate():
            # Large value for mate score
            return 10000.0 if score.mate() > 0 else -10000.0
        else:
            return float(score.cp)
    except Exception as e:
        print("Stockfish eval error:", e)
        return 0.0


# ============================================
# Board → Tensor Conversion (from Board)
# ============================================

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Converts a python-chess board into a (8, 8, 12) tensor.
    Planes:
        0..5  = white pawn, knight, bishop, rook, queen, king
        6..11 = black pawn, knight, bishop, rook, queen, king
    """
    tensor = np.zeros((8, 8, 12), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)   # 0..7
        file = chess.square_file(square)   # 0..7

        row = 7 - rank                     # row 0 is rank 8
        col = file

        channel = piece.piece_type - 1     # 0..5
        if piece.color == chess.BLACK:
            channel += 6

        tensor[row, col, channel] = 1.0

    return tensor


# ============================================
# Cached NN Evaluation (by piece placement only)
# ============================================

@lru_cache(maxsize=50000)
def nn_eval_cached(board_fen_only: str) -> float:
    """
    Neural network evaluation cached by FEN piece placement (board.board_fen()).
    Returns a scalar in [-1, 1] from White's perspective.
    """
    if not USE_NN or NN_MODEL is None:
        return 0.0

    # Rebuild a board from piece placement only
    board = chess.Board()
    board.set_board_fen(board_fen_only)

    tensor = board_to_tensor(board)
    tensor = np.expand_dims(tensor, axis=0)  # (1, 8, 8, 12)

    v = NN_MODEL.predict(tensor, verbose=0)
    return float(v[0][0])  # [-1, 1], White POV


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


def draw_eval_bar(screen, eval_cp: float):
    """
    Draw a vertical eval bar on the right side of the board.
    - White advantage: more white at the bottom.
    - Black advantage: more black at the bottom.
    eval_cp is in centipawns from White's perspective.
    """
    # Clamp eval to a range, e.g. [-1000, 1000] = [-10, +10] pawns
    max_cp = 1000.0
    clamped = max(-max_cp, min(max_cp, eval_cp))

    # t in [0, 1] -> fraction for White at bottom
    t = (clamped + max_cp) / (2.0 * max_cp)

    white_height = int(HEIGHT * t)
    x0 = BOARD_SIZE

    # Background black
    pygame.draw.rect(screen, (0, 0, 0), (x0, 0, EVAL_PANEL_WIDTH, HEIGHT))
    # White area at bottom
    pygame.draw.rect(screen, (255, 255, 255),
                     (x0, HEIGHT - white_height, EVAL_PANEL_WIDTH, white_height))

    # Middle line
    pygame.draw.line(screen, (128, 128, 128),
                     (x0, HEIGHT // 2), (x0 + EVAL_PANEL_WIDTH, HEIGHT // 2), 1)

    # Eval text
    if FONT is not None:
        # Convert cp to pawns
        pawns = eval_cp / 100.0
        text_str = f"{pawns:+.2f}"
        text_surf = FONT.render(text_str, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=(x0 + EVAL_PANEL_WIDTH // 2, 20))
        screen.blit(text_surf, text_rect)


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
    Static evaluation with positional + tactical heuristics.
    Returns score from side-to-move perspective.
    """

    if board.is_checkmate():
        return -100000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0

    PIECE_VALUE = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }

    # Pawn structure trackers
    pawn_files_white = {}
    pawn_files_black = {}

    # Track attacked and defended squares
    attacked_by_white = [False] * 64
    attacked_by_black = [False] * 64
    defended_by_white = [False] * 64
    defended_by_black = [False] * 64

    for sq in range(64):
        attacked_by_white[sq] = board.is_attacked_by(chess.WHITE, sq)
        attacked_by_black[sq] = board.is_attacked_by(chess.BLACK, sq)

        defended_by_white[sq] = board.is_attacked_by(chess.WHITE, sq)
        defended_by_black[sq] = board.is_attacked_by(chess.BLACK, sq)

    def piece_value(p): return PIECE_VALUE[p.piece_type]

    # =============================================
    # 1. MATERIAL + POSITIONAL FEATURES
    # =============================================
    for square, piece in board.piece_map().items():
        color = 1 if piece.color == chess.WHITE else -1
        pt = piece.piece_type
        val = PIECE_VALUE[pt]

        # Material
        score += color * val

        # Center control
        if square in [chess.D4, chess.E4, chess.D5, chess.E5]:
            score += color * 20
        elif square in [
            chess.C3, chess.C4, chess.C5, chess.C6,
            chess.D3, chess.E3, chess.F3,
            chess.F4, chess.F5, chess.F6
        ]:
            score += color * 10

        # Pawn structure tracking
        if pt == chess.PAWN:
            f = chess.square_file(square)
            if piece.color == chess.WHITE:
                pawn_files_white.setdefault(f, 0)
                pawn_files_white[f] += 1
            else:
                pawn_files_black.setdefault(f, 0)
                pawn_files_black[f] += 1

        # Knight outpost
        if pt == chess.KNIGHT and not board.is_attacked_by(not piece.color, square):
            rank = chess.square_rank(square)
            if (piece.color == chess.WHITE and rank >= 4) or \
               (piece.color == chess.BLACK and rank <= 3):
                score += color * 25

        # Bishop pair
        if pt == chess.BISHOP:
            if len([p for p in board.piece_map().values()
                    if p.piece_type == chess.BISHOP and p.color == piece.color]) == 2:
                score += color * 15

        # Rooks on open file
        if pt == chess.ROOK:
            file = chess.square_file(square)
            pawn_in_file = any(
                board.piece_at(chess.square(file, r)) and
                board.piece_at(chess.square(file, r)).piece_type == chess.PAWN
                for r in range(8)
            )
            if not pawn_in_file:
                score += color * 20

    # =============================================
    # 2. PAWN STRUCTURE PENALTIES
    # =============================================
    doubled = 15
    isolated = 20

    for f, count in pawn_files_white.items():
        if count > 1:
            score -= doubled * (count - 1)
        if f - 1 not in pawn_files_white and f + 1 not in pawn_files_white:
            score -= isolated

    for f, count in pawn_files_black.items():
        if count > 1:
            score += doubled * (count - 1)
        if f - 1 not in pawn_files_black and f + 1 not in pawn_files_black:
            score += isolated

    # =============================================
    # 3. MOBILITY
    # =============================================
    my_moves = board.legal_moves.count()
    board.push(chess.Move.null())
    opp_moves = board.legal_moves.count()
    board.pop()

    score += 2 * (my_moves - opp_moves)

    # =============================================
    # 4. TACTICAL AWARENESS
    # =============================================
    for square, piece in board.piece_map().items():
        color = piece.color
        val = piece_value(piece)

        att = attacked_by_white if color == chess.WHITE else attacked_by_black
        defd = defended_by_white if color == chess.WHITE else defended_by_black

        enemy_att = attacked_by_black if color == chess.WHITE else attacked_by_white

        # 4.1 Hanging pieces (attacked but not defended)
        if enemy_att[square] and not defd[square]:
            score += (-val * 0.8) if color == chess.WHITE else (val * 0.8)

        # 4.2 Loose pieces (undefended)
        if not defd[square]:
            score += (-25) if color == chess.WHITE else (25)

        # 4.3 Pinned pieces (to king)
        king_sq = board.king(color)
        if board.is_pinned(color, square) and piece.piece_type != chess.KING:
            score += (-40) if color == chess.WHITE else (40)

        # 4.4 Overloaded defenders (simple heuristic)
        defended_targets = [
            sq for sq in range(64)
            if enemy_att[sq] and defd[sq]
        ]
        if len(defended_targets) >= 2:
            score += (-20) if color == chess.WHITE else (20)

        # 4.5 Simple fork-like penalty (very rough)
        if enemy_att[square] and not defd[square]:
            score += (-val * 0.3) if color == chess.WHITE else (val * 0.3)

        # 4.6 King attack pressure
        for ksq in [board.king(chess.WHITE), board.king(chess.BLACK)]:
            if board.is_attacked_by(piece.color, ksq):
                score += 5 * (1 if piece.color == chess.WHITE else -1)

    # =============================================
    # 5. KING SAFETY (SIMPLE)
    # =============================================
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)

    if wk in (chess.G1, chess.C1):
        score += 30
    if bk in (chess.G8, chess.C8):
        score -= 30

    if board.fullmove_number > 10:
        if wk == chess.E1:
            score -= 20
        if bk == chess.E8:
            score += 20

    # =============================================
    # FINAL: Return from side-to-move perspective
    # =============================================
    return score if board.turn == chess.WHITE else -score


def evaluate_static(board: chess.Board) -> int:
    return static_eval(board)


# ============================================
# Quiescence Search
# ============================================

def generate_capture_moves(board: chess.Board):
    """Yield only capture moves (and promotions that are captures)."""
    for move in board.legal_moves:
        if board.is_capture(move) or move.promotion:
            yield move


def quiescence(board: chess.Board, alpha: float, beta: float, depth: int) -> int:
    """
    Quiescence search:
    - Evaluate the current position (stand pat).
    - Only search through capture sequences (and promotions).
    - Limited by QUIESCENCE_DEPTH to avoid blowup.
    """
    stand_pat = evaluate_static(board)

    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    if depth <= 0:
        return alpha

    for move in generate_capture_moves(board):
        board.push(move)
        score = -quiescence(board, -beta, -alpha, depth - 1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


# ============================================
# Move Ordering
# ============================================

def ordered_moves(board: chess.Board):
    moves = list(board.legal_moves)

    def score_move(m: chess.Move) -> int:
        val = 0
        if board.is_capture(m):
            # Simple MVV-LVA approximation using piece values
            captured = board.piece_at(m.to_square)
            attacker = board.piece_at(m.from_square)
            if captured:
                val += 10_000 + PIECE_VALUES[captured.piece_type]
            if attacker:
                val -= PIECE_VALUES[attacker.piece_type] // 10
        if m.promotion:
            val += 5_000
        if m.to_square in CENTER:
            val += 500
        return val

    moves.sort(key=score_move, reverse=True)
    return moves


# ============================================
# Negamax Search with Quiescence
# ============================================

def negamax(board: chess.Board, depth: int, alpha: float, beta: float) -> int:
    global NODES_SEARCHED, NODES_PRUNED

    # Count this node
    NODES_SEARCHED += 1

    if depth == 0 or board.is_game_over():
        return quiescence(board, alpha, beta, QUIESCENCE_DEPTH)

    best = -math.inf

    for move in ordered_moves(board):
        board.push(move)
        val = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if val > best:
            best = val
        if val > alpha:
            alpha = val

        # Count pruned nodes
        if alpha >= beta:
            NODES_PRUNED += 1
            break

    return best


# ============================================
# AI Player (NN only at root)
# ============================================

class AIPlayer:
    def __init__(self, depth: int = 5):  # can change the depth
        self.depth = depth

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        """
        For each candidate move:
            1. Do a static-only search with quiescence (negamax).
            2. Evaluate resulting position with the NN once (optional).
            3. Blend static + NN scores at the ROOT ONLY.
        """
        best_move = None
        best_score = -math.inf

        ai_color = board.turn  # side the engine is playing

        for move in ordered_moves(board):
            board.push(move)

            # Static + quiescence search score from AI perspective
            static_score = -negamax(board, self.depth - 1, -math.inf, math.inf)

            blended_score = static_score

            if USE_NN:
                global NN_EVAL_COUNT
                NN_EVAL_COUNT += 1
                fen_key = board.board_fen()
                nn_raw = nn_eval_cached(fen_key)     # [-1, 1] from White POV
                nn_cp = nn_raw * 2000.0              # to centipawns

                if ai_color == chess.WHITE:
                    nn_side = nn_cp
                else:
                    nn_side = -nn_cp

                blended_score = int(0.7 * static_score + 0.3 * nn_side)

            board.pop()

            if blended_score > best_score:
                best_score = blended_score
                best_move = move

        return best_move


# ============================================
# Main Loop
# ============================================

def main():
    global NODES_SEARCHED, NODES_PRUNED, NN_EVAL_COUNT, CURRENT_EVAL_CP, FONT

    pygame.init()
    pygame.font.init()
    FONT = pygame.font.SysFont("arial", 18)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess AI + NN + Quiescence + Eval Bar")

    board = chess.Board()
    load_images()

    ai = AIPlayer(depth=3)

    selected = None
    promoting = False
    promo_from = promo_to = None
    promo_boxes = []

    # Initial eval
    CURRENT_EVAL_CP = evaluate_with_stockfish(board)

    clock = pygame.time.Clock()
    running = True

    while running:
        # You can uncap or cap FPS as you like
        # clock.tick(60)

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
                            # Update eval after a successful promotion move
                            CURRENT_EVAL_CP = evaluate_with_stockfish(board)

                        promoting = False
                        selected = None
                continue

            # Normal clicks
            if event.type == pygame.MOUSEBUTTONDOWN and not promoting:
                mx, my = event.pos

                # Ignore clicks in the eval bar panel
                if mx >= BOARD_SIZE:
                    continue

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
                            # Update eval after human move
                            CURRENT_EVAL_CP = evaluate_with_stockfish(board)
                        selected = None

        # AI move
        if not promoting and board.turn == chess.BLACK and not board.is_game_over():
            mv = ai.choose_move(board)
            if mv:
                board.push(mv)

                # Update eval after AI move
                CURRENT_EVAL_CP = evaluate_with_stockfish(board)

                # Print metrics for debugging
                print(f"Nodes searched: {NODES_SEARCHED:,}")
                print(f"Nodes pruned:   {NODES_PRUNED:,}")
                print(f"NN eval count:  {NN_EVAL_COUNT:,}")
                print("-" * 40)

                # Reset counters for next move
                NODES_SEARCHED = 0
                NODES_PRUNED = 0
                NN_EVAL_COUNT = 0

        # Draw
        if not promoting:
            # Clear only the chessboard area
            draw_board(screen)
            highlight_squares(screen, board, selected)
            draw_pieces(screen, board)

            # Draw eval bar on the right
            draw_eval_bar(screen, CURRENT_EVAL_CP)

            pygame.display.flip()

    pygame.quit()

    # Cleanly shut down Stockfish
    if SF_ENGINE is not None:
        try:
            SF_ENGINE.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
