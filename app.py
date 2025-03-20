import glob
from flask import Flask, send_file, send_from_directory
import os

from parse_lark import trees_dir

app = Flask(__name__)


@app.route('/')
def serve_doctor():
    return send_from_directory('src2', 'doctor.html')


@app.route('/load_game_auto')
def load_game_auto():
    tree_paths = glob.glob(os.path.join(trees_dir, '*'))
    trees = []
    tree_paths = sorted(tree_paths, reverse=True)
    test_games = ['sokoban_basic']
    test_game_paths = [os.path.join(trees_dir, tg + '.pkl') for tg in test_games]
    tree_paths = test_game_paths + tree_paths


if __name__ == "__main__":
    app.run(debug=True)
