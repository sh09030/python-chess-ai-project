import time
import numpy as np
import chess
from tensorflow.keras.models import load_model

# ============================================
# Load your model
# ============================================

model = load_model("eval_net.h5", compile=False)

# ============================================
# Your real board_to_tensor function
# ============================================

def board_to_tensor(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((8, 8, 12), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)

        row = 7 - rank
        col = file

        channel = piece.piece_type - 1
        if piece.color == chess.BLACK:
            channel += 6

        tensor[row, col, channel] = 1.0

    return tensor


# ============================================
# A realistic board position (replace as needed)
# ============================================

fen = "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 4"
board = chess.Board(fen)

# ============================================
# Create tensor from real board
# ============================================

tensor = board_to_tensor(board)
tensor = np.expand_dims(tensor, axis=0)  # (1, 8, 8, 12)

# Warm-up runs (important!)
for _ in range(5):
    model.predict(tensor, verbose=0)


# ============================================
# Measure pure forward pass time
# ============================================

start = time.time()
out = model.predict(tensor, verbose=0)
end = time.time()

forward_ms = (end - start) * 1000
print("Forward pass only:", forward_ms, "ms")


# ============================================
# Measure full cost: board → tensor → NN
# ============================================

N = 100
start = time.time()

for _ in range(N):
    t = board_to_tensor(board)
    t = np.expand_dims(t, axis=0)
    model.predict(t, verbose=0)

end = time.time()

full_avg_ms = ((end - start) / N) * 1000

print("Full eval (board_to_tensor + forward):", full_avg_ms, "ms")
print("Total time for", N, "runs:", end - start, "seconds")
