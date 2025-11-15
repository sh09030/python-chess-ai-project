# chess_engine.py
# Phase 1: rules-correct chess with human vs. random AI, CLI interface.

import sys
import random
import chess
import chess.pgn
from typing import Optional

HELP = """
Commands:
  - Move: enter either UCI (e2e4, g1f3, e7e8q) or SAN (Nf3, exd5, O-O, Qxe6+)
  - moves        show all legal moves for the side to move
  - undo         undo the last full move pair if possible (or one move if only one exists)
  - fen          print current FEN
  - pgn          print PGN so far
  - help         show this help
  - quit         exit
"""

def print_board(board: chess.Board) -> None:
    # Unicode board with pieces, ranks, files
    print(board.unicode(borders=True))
    turn = "White" if board.turn == chess.WHITE else "Black"
    print(f"Turn: {turn}  |  Move #: {board.fullmove_number}")
    if board.is_check():
        print("Check!")

def parse_move(user_input: str, board: chess.Board) -> Optional[chess.Move]:
    """Accept SAN or UCI. Return a legal chess.Move or None."""
    s = user_input.strip()
    if not s:
        return None

    # Try SAN first
    try:
        move = board.parse_san(s)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass

    # Try UCI next
    try:
        move = chess.Move.from_uci(s)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass

    return None

def result_string(board: chess.Board) -> str:
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate. {winner} wins."
    if board.is_stalemate():
        return "Draw by stalemate."
    if board.is_insufficient_material():
        return "Draw by insufficient material."
    if board.can_claim_threefold_repetition():
        return "Draw by threefold repetition."
    if board.can_claim_fifty_moves():
        return "Draw by fifty-move rule."
    if board.is_fivefold_repetition():
        return "Draw by fivefold repetition."
    if board.is_seventyfive_moves():
        return "Draw by seventy-five move rule."
    return "Game over."

class RandomAI:
    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        if not moves:
            return None
        return random.choice(moves)

class Game:
    def __init__(self, human_plays_white: bool = True, rng_seed: Optional[int] = None):
        self.board = chess.Board()
        self.ai = RandomAI()
        self.human_white = human_plays_white
        if rng_seed is not None:
            random.seed(rng_seed)
        self.game = chess.pgn.Game()
        self.node = self.game

    def add_move_to_pgn(self, move: chess.Move) -> None:
        self.node = self.node.add_variation(move)

    def undo(self) -> None:
        # Undo one ply. If you want full move pair, call twice.
        if len(self.board.move_stack) == 0:
            print("Nothing to undo.")
            return
        self.board.pop()
        # Rebuild PGN node from scratch after undo to keep it consistent.
        self.rebuild_pgn()

    def rebuild_pgn(self) -> None:
        self.game = chess.pgn.Game()
        self.node = self.game
        replay = chess.Board()
        for mv in self.board.move_stack:
            self.node = self.node.add_variation(mv)
            replay.push(mv)

    def print_status_if_over(self) -> bool:
        if self.board.is_game_over():
            print(result_string(self.board))
            return True
        return False

    def play(self) -> None:
        print("Python Chess Engine: Human vs Random AI")
        print("Type 'help' for commands.")
        print_board(self.board)

        while True:
            # Decide whose turn it is
            human_to_move = (self.board.turn == chess.WHITE and self.human_white) or \
                            (self.board.turn == chess.BLACK and not self.human_white)

            if human_to_move:
                user_input = input("> ").strip().lower()

                if user_input == "quit":
                    print("Bye.")
                    break
                elif user_input == "help":
                    print(HELP)
                    continue
                elif user_input == "moves":
                    legal = sorted([m.uci() for m in self.board.legal_moves])
                    print(f"Legal moves ({len(legal)}): {' '.join(legal)}")
                    continue
                elif user_input == "fen":
                    print(self.board.fen())
                    continue
                elif user_input == "pgn":
                    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
                    print(self.game.accept(exporter))
                    continue
                elif user_input == "undo":
                    # Undo one ply. If AI moved last, undo twice to return to human.
                    self.undo()
                    print_board(self.board)
                    continue
                elif user_input == "":
                    continue

                move = parse_move(user_input, self.board)
                if move is None:
                    print("Invalid or illegal move. Use UCI (e2e4) or SAN (Nf3). Type 'moves' to see options.")
                    continue

                self.board.push(move)
                self.add_move_to_pgn(move)
                print_board(self.board)
                if self.print_status_if_over():
                    break
            else:
                # AI move
                move = self.ai.choose_move(self.board)
                if move is None:
                    # No legal moves
                    print_board(self.board)
                    if self.print_status_if_over():
                        break
                    else:
                        print("AI has no legal moves. Something unexpected happened.")
                        break

                print(f"AI plays: {self.board.san(move)} ({move.uci()})")
                self.board.push(move)
                self.add_move_to_pgn(move)
                print_board(self.board)
                if self.print_status_if_over():
                    break

def choose_side() -> bool:
    while True:
        side = input("Play as White or Black? [w/b]: ").strip().lower()
        if side in ("w", "white", ""):
            return True
        if side in ("b", "black"):
            return False
        print("Please enter w or b.")

if __name__ == "__main__":
    human_white = choose_side()
    # Optional reproducibility: pass a seed, e.g., rng_seed=42
    game = Game(human_plays_white=human_white, rng_seed=None)
    try:
        game.play()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
        sys.exit(0)
