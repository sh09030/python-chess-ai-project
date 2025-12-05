import chess
import chess.pgn
import numpy as np
from stockfish import Stockfish
import random
import time

# Path to your Stockfish binary
STOCKFISH_PATH = "/usr/local/bin/stockfish"

stockfish = Stockfish(STOCKFISH_PATH)
stockfish.update_engine_parameters({
    "Threads": 4,
    "Hash": 512
})

# -----------------------------------------------
# Convert Board → (8, 8, 12) tensor
# -----------------------------------------------
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
        idx = piece_map[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
        tensor[row, col, idx] = 1.0

    return tensor


# -----------------------------------------------
# Evaluate a position deeply with Stockfish
# -----------------------------------------------
def sf_eval(board, depth=14):
    stockfish.set_fen_position(board.fen())
    info = stockfish.get_evaluation()

    if info["type"] == "mate":
        return 10000 if info["value"] > 0 else -10000

    return info["value"]  # centipawns


# -----------------------------------------------
# Engine vs Engine game generator
# -----------------------------------------------
def play_engine_game(max_moves=120):
    board = chess.Board()

    # Let Stockfish pick moves for both sides
    while not board.is_game_over() and board.fullmove_number <= max_moves:
        stockfish.set_fen_position(board.fen())

        move = stockfish.get_best_move()
        if move is None:
            break

        board.push(chess.Move.from_uci(move))

        yield board.copy()  # yield after every move


# -----------------------------------------------
# Main Dataset Generator (Engine vs Engine)
# -----------------------------------------------
def generate_e2e_dataset(N=20000, eval_depth=14, sample_every=1):
    X = []
    y = []

    print(f"Generating {N} engine vs engine positions...")
    start_time = time.time()

    while len(X) < N:
        for pos in play_engine_game():
            # sample only every nth move if you want to reduce correlation
            if (pos.fullmove_number % sample_every) != 0:
                continue

            tensor = board_to_tensor(pos)
            eval_cp = sf_eval(pos, depth=eval_depth)

            X.append(tensor)
            y.append(eval_cp)

            if len(X) % 500 == 0 or len(X) == N:
                elapsed = time.time() - start_time
                done = len(X)
                pct = 100.0 * done / N
                pos_per_sec = done / elapsed if elapsed > 0 else 0.0
                remaining = (N - done) / pos_per_sec if pos_per_sec > 0 else 0.0

                # Print a single updating progress line
                print(
                    f"\r{done}/{N} positions "
                    f"({pct:.2f} percent)  "
                    f"{pos_per_sec:.2f} pos/sec  "
                    f"ETA {remaining/60:.1f} min",
                    end="",
                    flush=True
                )

            if len(X) >= N:
                break

    print()  # newline after the progress line

    X = np.array(X)
    y = np.array(y)

    np.save("x_positions.npy", X)
    np.save("y_evals.npy", y)

    total_time = time.time() - start_time
    print("\nSaved x_positions.npy and y_evals.npy")
    print(f"Dataset generation complete in {total_time/60:.2f} minutes")


if __name__ == "__main__":
    start = time.time()
    generate_e2e_dataset(
        N=5000,        # small test
        eval_depth=14,
        sample_every=1
    )
    end = time.time()
    elapsed = end - start
    print(f"Time for 5000 positions: {elapsed:.2f} sec")
    print(f"Per position: {elapsed / 5000:.4f} sec")
    print(f"Estimated time for 1,000,000: {(elapsed / 5000) * 1_000_000 / 3600:.2f} hours")
