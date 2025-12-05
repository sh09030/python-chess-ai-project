import math
import chess
import numpy as np

from functools import lru_cache

import tensorflow as tf
from tensorflow.keras.models import load_model

# ============================================
# Engine Metrics (optional, for debugging)
# ============================================

NODES_SEARCHED = 0
NODES_PRUNED = 0
NN_EVAL_COUNT = 0

# Quiescence depth limit to avoid explosion
QUIESCENCE_DEPTH = 4

# ============================================
# TensorFlow: Configure (CPU-only is fine)
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
    Returns score from side-to-move perspective (centipawns).
    Positive is good for side to move.
    """

    # Terminal conditions
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

    # Track attacked and "defended" squares
    attacked_by_white = [False] * 64
    attacked_by_black = [False] * 64
    defended_by_white = [False] * 64
    defended_by_black = [False] * 64

    for sq in range(64):
        attacked_by_white[sq] = board.is_attacked_by(chess.WHITE, sq)
        attacked_by_black[sq] = board.is_attacked_by(chess.BLACK, sq)

        # Defense = attacked by same color
        defended_by_white[sq] = attacked_by_white[sq]
        defended_by_black[sq] = attacked_by_black[sq]

    def piece_value(p): 
        return PIECE_VALUE[p.piece_type]

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
        if square in (chess.D4, chess.E4, chess.D5, chess.E5):
            score += color * 20
        elif square in (
            chess.C3, chess.C4, chess.C5, chess.C6,
            chess.D3, chess.E3, chess.F3,
            chess.F4, chess.F5, chess.F6
        ):
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

        # Knight outpost: advanced, not attacked by enemy pawn/ piece
        if pt == chess.KNIGHT and not board.is_attacked_by(not piece.color, square):
            rank = chess.square_rank(square)
            if (piece.color == chess.WHITE and rank >= 4) or \
               (piece.color == chess.BLACK and rank <= 3):
                score += color * 25

        # Bishop pair bonus
        if pt == chess.BISHOP:
            if len([p for p in board.piece_map().values()
                    if p.piece_type == chess.BISHOP and p.color == piece.color]) == 2:
                score += color * 15

        # Rooks on open file
        if pt == chess.ROOK:
            file = chess.square_file(square)
            pawn_in_file = any(
                (board.piece_at(chess.square(file, r)) is not None) and
                (board.piece_at(chess.square(file, r)).piece_type == chess.PAWN)
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
        color = piece.color       # True = WHITE, False = BLACK
        val = piece_value(piece)

        att = attacked_by_white if color == chess.WHITE else attacked_by_black
        defd = defended_by_white if color == chess.WHITE else defended_by_black

        enemy_att = attacked_by_black if color == chess.WHITE else attacked_by_white
        enemy_def = defended_by_black if color == chess.WHITE else defended_by_white

        # 4.1 Hanging pieces (attacked but not defended)
        if enemy_att[square] and not defd[square]:
            score += (-val * 0.8) if color == chess.WHITE else (val * 0.8)

        # 4.2 Loose pieces (undefended at all)
        if not defd[square]:
            score += (-25) if color == chess.WHITE else (25)

        # 4.3 Pinned pieces (to own king)
        if board.is_pinned(color, square) and piece.piece_type != chess.KING:
            score += (-40) if color == chess.WHITE else (40)

        # 4.4 Overloaded defenders: same defender covering many attacked squares
        defended_targets = [
            sq for sq in range(64)
            if enemy_att[sq] and defd[sq]
        ]
        if len(defended_targets) >= 2:
            score += (-20) if color == chess.WHITE else (20)

        # 4.5 Simple fork: attacked more than defended (very rough)
        # NOTE: enemy_att[square] and defd[square] are booleans in this version,
        # so this part is mostly a placeholder for richer attacker counts if added later.
        if enemy_att[square] and not defd[square]:
            score += (-val * 0.3) if color == chess.WHITE else (val * 0.3)

        # 4.6 King attack pressure
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        for ksq in (wk, bk):
            if ksq is not None and board.is_attacked_by(piece.color, ksq):
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
    """
    Negamax with alpha-beta pruning.
    Uses static eval + quiescence search at leaf nodes.
    Neural net is NOT called here to keep it fast.
    """
    global NODES_SEARCHED, NODES_PRUNED

    NODES_SEARCHED += 1   # Count this node

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

        if alpha >= beta:
            NODES_PRUNED += 1  # beta cutoff
            break

    return best

# ============================================
# AI Player (NN only at root)
# ============================================

class AIPlayer:
    def __init__(self, depth: int = 3):
        self.depth = depth

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        """
        For each candidate move:
            1. Do a static-only search with quiescence (negamax).
            2. Evaluate resulting position with the NN once (optional).
            3. Blend static + NN scores at the ROOT ONLY.
        """
        global NN_EVAL_COUNT

        best_move = None
        best_score = -math.inf

        ai_color = board.turn  # side the engine is playing

        for move in ordered_moves(board):
            board.push(move)

            # Static + quiescence search score from AI perspective
            static_score = -negamax(board, self.depth - 1, -math.inf, math.inf)

            blended_score = static_score

            if USE_NN:
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
