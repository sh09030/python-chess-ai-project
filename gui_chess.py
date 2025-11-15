import math
import pygame
import chess

WIDTH = 640
HEIGHT = 640
SQ_SIZE = WIDTH // 8

IMAGES = {}

# ------------------------------
# Load images
# ------------------------------

def load_images():
    pieces = ["wp", "wn", "wb", "wr", "wq", "wk",
              "bp", "bn", "bb", "br", "bq", "bk"]
    for p in pieces:
        IMAGES[p] = pygame.transform.scale(
            pygame.image.load(f"pieces/{p}.png"),
            (SQ_SIZE, SQ_SIZE)
        )

# ------------------------------
# Drawing
# ------------------------------

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

    # selected square
    col = chess.square_file(selected_square)
    row = 7 - chess.square_rank(selected_square)

    s = pygame.Surface((SQ_SIZE, SQ_SIZE))
    s.set_alpha(120)
    s.fill((246, 246, 105))  # light yellow
    screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))

    # legal moves from that square
    for move in board.legal_moves:
        if move.from_square == selected_square:
            dest_c = chess.square_file(move.to_square)
            dest_r = 7 - chess.square_rank(move.to_square)
            center = (dest_c * SQ_SIZE + SQ_SIZE // 2,
                      dest_r * SQ_SIZE + SQ_SIZE // 2)
            pygame.draw.circle(screen, (50, 50, 50), center, 12)

# ------------------------------
# Evaluation
# ------------------------------

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # king safety handled via game termination
}

CENTER_SQUARES = {
    chess.D4, chess.E4, chess.D5, chess.E5
}
EXTENDED_CENTER = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4,
    chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6
}

def evaluate_board(board: chess.Board) -> int:
    """
    Static eval from the perspective of the side to move.
    Positive is good for the side to move, negative is bad.
    Units are "centipawns".
    """

    # terminal positions
    if board.is_checkmate():
        # side to move is checkmated, so very bad
        return -10_000
    if board.is_stalemate() or board.is_insufficient_material() \
       or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0

    material_white = 0
    material_black = 0
    center_bonus_white = 0
    center_bonus_black = 0

    for square, piece in board.piece_map().items():
        val = PIECE_VALUES[piece.piece_type]

        # center control: reward pieces on central squares
        center_bonus = 0
        if square in CENTER_SQUARES:
            center_bonus = 20
        elif square in EXTENDED_CENTER:
            center_bonus = 10

        if piece.color == chess.WHITE:
            material_white += val
            center_bonus_white += center_bonus
        else:
            material_black += val
            center_bonus_black += center_bonus

    material_score = material_white - material_black
    center_score = center_bonus_white - center_bonus_black

    # mobility: difference in legal moves
    mobility_score = 0
    side_to_move_legal = board.legal_moves.count()
    board.push(chess.Move.null())
    opponent_legal = board.legal_moves.count()
    board.pop()
    mobility_score = 5 * (side_to_move_legal - opponent_legal)

    total_from_white = material_score + center_score + mobility_score

    if board.turn == chess.WHITE:
        return total_from_white
    else:
        return -total_from_white

# ------------------------------
# Negamax + alpha beta
# ------------------------------

def ordered_moves(board: chess.Board):
    """
    Simple move ordering: captures and promotions first.
    """
    moves = list(board.legal_moves)

    def score_move(m: chess.Move):
        score = 0
        if board.is_capture(m):
            score += 10_000
        if m.promotion is not None:
            score += 5_000
        # small bonus for moves towards the center
        to_sq = m.to_square
        if to_sq in CENTER_SQUARES:
            score += 500
        elif to_sq in EXTENDED_CENTER:
            score += 200
        return score

    moves.sort(key=score_move, reverse=True)
    return moves

def negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    """
    Negamax search with alpha beta pruning.
    Returns a score from the perspective of the side to move.
    """
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    max_score = -math.inf

    for move in ordered_moves(board):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if score > max_score:
            max_score = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    return max_score

# ------------------------------
# AI Player
# ------------------------------

class AIPlayer:
    def __init__(self, depth: int = 3):
        self.depth = depth

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        best_move = None
        best_score = -math.inf

        for move in ordered_moves(board):
            board.push(move)
            score = -negamax(board, self.depth - 1, -math.inf, math.inf)
            board.pop()

            if score > best_score or best_move is None:
                best_score = score
                best_move = move

        return best_move

# ------------------------------
# Main Game Loop
# ------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Python Chess GUI")

    board = chess.Board()
    load_images()

    # human plays White, AI plays Black
    human_plays_white = True
    ai = AIPlayer(depth=3)

    selected_square = None
    running = True

    clock = pygame.time.Clock()

    while running:
        clock.tick(60)  # limit FPS a bit

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # human move (white)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if board.turn == chess.WHITE and human_plays_white and not board.is_game_over():
                    x, y = event.pos
                    row = y // SQ_SIZE
                    col = x // SQ_SIZE
                    square = chess.square(col, 7 - row)

                    if selected_square is None:
                        # only allow selecting your own pieces
                        piece = board.piece_at(square)
                        if piece is not None and piece.color == chess.WHITE:
                            selected_square = square
                    else:
                        move = chess.Move(selected_square, square)
                        if move in board.legal_moves:
                            board.push(move)
                        selected_square = None

        # AI move (black)
        if not board.is_game_over():
            if board.turn == chess.BLACK and human_plays_white:
                ai_move = ai.choose_move(board)
                if ai_move is not None:
                    board.push(ai_move)

        draw_board(screen)
        highlight_squares(screen, board, selected_square)
        draw_pieces(screen, board)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
