`source .venv/bin/activate`

# Environment setup (NCA world model)

The repo's `README.md` "Setup" section is sparse. Concrete steps that work
on a fresh box:

```bash
# 1. Submodules — PuzzleScript is required (tokenizer reads its colors.js).
git submodule update --init PuzzleScript

# 2. Python 3.13 venv via uv (system python may be older).
uv venv --python 3.13 .venv

# 3. Install requirements.
VIRTUAL_ENV=$(pwd)/.venv uv pip install -r requirements.txt

# 4. requirements.txt ships CPU JAX (the jax[cuda] line is commented out).
#    On a CUDA box, upgrade explicitly to match the pinned version:
VIRTUAL_ENV=$(pwd)/.venv uv pip install -U "jax[cuda12]==0.7.1"

# 5. pybind11 is required by setup_cpp.py but not in requirements.txt.
VIRTUAL_ENV=$(pwd)/.venv uv pip install pybind11

# 6. Build the C++ engine extension.
.venv/bin/python setup_cpp.py build_ext --inplace

# 7. Sanity check.
.venv/bin/python -c "
from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
from nca_wm.tokenize_game import VOCAB_SIZE_BASE
import jax
print('jax', jax.__version__, jax.devices())
print('VOCAB_SIZE_BASE', VOCAB_SIZE_BASE)
"
```

# Data collection (`nca_wm/`)

Rollout caches live under `rollout_data/{game}/level_{i}/*.npz` and are
gitignored — every machine has to populate them itself.

Important: `parallel_collect.py --games <preset>` only recognises the
literal string `gallery`. Multi-game presets defined in
`MULTI_GAME_PRESETS` (e.g. `small`, `scaling_14`) are NOT expanded — pass
the games as a comma-separated list instead. Example for the `small`
preset (9 games):

```bash
.venv/bin/python -m nca_wm.scripts.parallel_collect \
    --games "nekopuzzle,notsnake,blocks,sokoban_basic,sokoban_match3,Zen_Puzzle_Garden,Multi-word_Dictionary_Game,kettle,Travelling_salesman" \
    --workers 8
```

`small` collection takes ~15 min on 8 CPU workers and produces ~300 MB of
caches. `kettle` is the long pole (~14 min).

# Known gotchas

- `nca_wm/scripts/parallel_collect.py:26` hardcodes
  `sys.path.insert(0, "/home/jupyter-smearle/script-doctor")`. Harmless on
  other machines (the path doesn't exist, the import still resolves via
  `python -m`) but worth knowing if you ever see "smearle" in a stack
  trace.
- `nca_wm/train.py` `--n_random_episodes 0` crashes in
  `collect_random_rollouts` (reads `cached["ep_ends"]` when there is no
  cache). Use the default (500) or any positive value.
- `requirements.txt` line for jax/jax[cuda] is commented in a confusing
  way; ignore the comment and follow steps 4–5 above.
- For `data/scraped_games/` to contain the games you need, `git lfs` is
  not required — they're plain `.txt` files committed to the repo (~952
  scraped + ~100 custom).
