"""PuzzleScript grammar construction without JAX or generation clients."""
from lark import Lark
from puzzlescript_jax.globals import LARK_SYNTAX_PATH


def init_ps_lark_parser():
    with open(LARK_SYNTAX_PATH, "r", encoding='utf-8') as file:
        puzzlescript_grammar = file.read()
    # Initialize the Lark parser with the PuzzleScript grammar
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    return parser
