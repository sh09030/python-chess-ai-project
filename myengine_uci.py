import sys
import chess
from engine_core import AIPlayer

class UCIEngine:
    def __init__(self):
        self.board = chess.Board()
        self.ai = AIPlayer(depth=3)

    def send(self, msg):
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()

    def uci(self):
        self.send("id name FarazNN")
        self.send("id author Faraz")
        self.send("uciok")

    def isready(self):
        self.send("readyok")

    def ucinewgame(self):
        self.board = chess.Board()

    def position(self, line):
        parts = line.split()

        if parts[1] == "startpos":
            self.board = chess.Board()
            if "moves" in parts:
                idx = parts.index("moves") + 1
                for move in parts[idx:]:
                    self.board.push_uci(move)

        elif parts[1] == "fen":
            # FEN is 6 fields
            fen = " ".join(parts[2:8])
            self.board = chess.Board(fen)

            if "moves" in parts:
                idx = parts.index("moves") + 1
                for move in parts[idx:]:
                    self.board.push_uci(move)

    def go(self, line):
        move = self.ai.choose_move(self.board)
        if move:
            self.send("bestmove " + move.uci())
        else:
            self.send("bestmove 0000")

    def loop(self):
        while True:
            line = sys.stdin.readline().strip()
            if not line:
                continue

            if line == "uci":
                self.uci()
            elif line == "isready":
                self.isready()
            elif line.startswith("position"):
                self.position(line)
            elif line.startswith("ucinewgame"):
                self.ucinewgame()
            elif line.startswith("go"):
                self.go(line)
            elif line == "quit":
                break

if __name__ == "__main__":
    UCIEngine().loop()
