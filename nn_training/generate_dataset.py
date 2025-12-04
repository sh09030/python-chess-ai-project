import chess
import numpy as np
import random
from stockfish import Stockfish


STOCKFISH_PATH = "/usr/local/bin/stockfish"

stockfish = Stockfish(STOCKFISH_PATH)
stockfish.update_engine_parameters({"Threads": 4, "Hash": 512})

def board_to_tensor(board):
    tensor = np.zeros((8, 8, 12), dtype=np.float32)

    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        base = piece_map[piece.piece_type]
        offset = 0 if piece.color == chess.WHITE else 6

        tensor[row, col, base + offset] = 1
    
    return tensor


def random_position():
    board = chess.Board()
    moves = random.randint(10, 40)

    for _ in range(moves):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)

    return board


def stockfish_eval(board, depth=12):
    stockfish.set_fen_position(board.fen())
    info = stockfish.get_evaluation()

    if info["type"] == "mate":
        return 10000 if info["value"] > 0 else -10000
    return info["value"]  # centipawns


def generate_dataset(N=20000):
    X = []
    y = []

    for i in range(N):
        board = random_position()
        tensor = board_to_tensor(board)
        eval_cp = stockfish_eval(board)

        X.append(tensor)
        y.append(eval_cp)

        if i % 200 == 0:
            print(f"{i}/{N} positions done")

    np.save("x_positions.npy", np.array(X))
    np.save("y_evals.npy", np.array(y))
    print("Dataset saved!")


if __name__ == "__main__":
    generate_dataset(20000)
