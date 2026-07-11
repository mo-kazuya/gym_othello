# CLAUDE.md

Guidance for AI assistants (Claude Code) working in this repository.

## What this is

`gym_othello` is a small Python package implementing an Othello (Reversi)
board game as a [Gymnasium](https://gymnasium.farama.org/)-style
multi-agent environment (two agents: `"white"` and `"black"`), intended for
use with RL frameworks such as Ray RLlib. It also supports interactive human
play via `pygame`.

The entire implementation lives in one file:
`gym_othello/envs/othello_env.py`. The rest of the repo is packaging
scaffolding, a single unit test, and a demo notebook.

## Repository layout

```
gym_othello/
  __init__.py            # currently empty
  envs/
    __init__.py           # exports OthelloEnv
    othello_env.py         # Board, Player, OthelloEnv — all game logic lives here
    atari.ttf               # font used for on-screen score/status text (pygame render)
example/
  example1.ipynb           # demo: random agent vs. human (mouse) agent, render_mode='human'
  train_dqn.py             # demo: DQN self-play training (PyTorch); --eval-only evaluates vs random
  dqn_othello.pt           # trained DQN weights (400 episodes, ~77% win rate vs random)
  train_wthor.py           # demo: supervised policy training from WTHOR game records
  policy_wthor.pt          # policy net trained by train_wthor.py (see file header for data setup)
  train_ppo_transformer.py # demo: PPO self-play training of a Transformer policy
  ppo_transformer_othello.pt # trained Transformer weights from train_ppo_transformer.py
  wthor_data/              # (gitignored) .wtb files — real ones from ffothello.org, or
                           # synthetic ones via train_wthor.py --make-synthetic N
  heuristic_agent.py       # demo: non-DNN agent — positional-weight table + minimax/alpha-beta
                           # (no training; beats all three trained models above at depth=3)
  transformer_minimax_agent.py # demo: AlphaZero-style hybrid — minimax/alpha-beta search using
                           # the trained Transformer's value head as leaf evaluator and its
                           # policy head for move ordering (see file header: still loses cleanly
                           # to heuristic_agent.py's minimax at matched depth — search alone
                           # doesn't fix an undertrained leaf evaluator)
test/
  test_a.py                 # unittest-based tests for Board and OthelloEnv
setup.py                    # packaging metadata (no install_requires declared)
MANIFEST.in                 # ensures atari.ttf is bundled in the package
README.md                    # Japanese install/usage instructions
```

## Key code (`gym_othello/envs/othello_env.py`)

- **`Board`** — pure game logic: 8x8 board as `list[list[int]]`, cells are
  `0` (empty), `PLAYER_WHITE = 1`, `PLAYER_BLACK = 2`.
  - `initialize_board()` sets up the standard 4-piece starting position.
  - `is_valid_move(col, row, player)` / `get_valid_moves(player)` — legality
    checks (note the coordinate order: methods take `(col, row)`, but the
    board is indexed `self.board[row][col]`).
  - `place_piece(col, row, player)` places a piece, flips captured pieces,
    and advances `current_player`, auto-skipping a player's turn if they
    have no valid moves (pass rule).
  - `draw_board(screen)` — pygame rendering (board, valid-move markers,
    score, win/lose/draw banner). Only used when `render_mode="human"`.
  - `is_game_over()` — true when neither player has a valid move.

- **`OthelloEnv(gym.Env)`** — multi-agent-flavored Gymnasium env:
  - `agents` / `possible_agents` are `["white", "black"]`.
  - `observation_spaces["<agent>"]` is a `Dict` with:
    - `observation`: `Box(0,1,(2,8,8))` — stacked `[my_board, opponent_board]`
      binary planes (perspective-relative, not absolute white/black).
    - `action_mask`: `Box(0,1,(8,8))` — 1 where a legal move exists.
  - `action_spaces["<agent>"]` is `Discrete(64)`; an action `pos` maps to
    board coords via `col = pos % 8`, `row = pos // 8`.
  - `step(action)` expects `action` to be a **dict** keyed by the current
    player's name, e.g. `{"white": 42}` — passing the wrong key (or the
    non-current player's key) ends the episode with a `-10.0` reward as an
    invalid-action penalty. An illegal move for the correct player does the
    same.
  - Corner moves (0,0 / 7,0 / 0,7 / 7,7) grant a `+0.5` / `-0.5` shaping
    reward to the mover/opponent in addition to any terminal reward.
  - Terminal reward on game end is `ceil(|count_diff| / 5)` magnitude,
    signed by which side has more pieces, and negated for the loser (so
    `rewards['white'] == -rewards['black']`, always zero-sum). This
    replaced an earlier `(count_diff + 4) // 5.0` formula whose floor
    division on a possibly-negative numerator was not zero-sum (e.g.
    `count_diff=+1` gave white `+1.0` / black `0.0` instead of a
    symmetric split) — fixed by taking `ceil` on the absolute value and
    re-applying the sign afterwards.
  - `step()` validates that the action integer is in `[0, 64)` before
    converting it to `(col, row)`; out-of-range actions are treated the
    same as any other invalid action (`-10.0` penalty, episode ends)
    instead of raising an `IndexError` or silently wrapping via Python's
    negative-list-indexing on a bad `row`.
  - `reset(seed=None, options=None)` supports a random opening offset: pass
    `options={"offset": N}`, or construct the env with `random_offset=N` to
    play N random half-moves before returning control (used to diversify
    starting states for training).
  - `render()` draws to the pygame window when `render_mode="human"`,
    otherwise returns the raw board (list of lists) — it does not follow
    the usual Gymnasium convention of returning an RGB array.
  - `random_agent(obs)` / `human_agent(obs)` are classmethods usable as
    simple opponents: `random_agent` samples from `action_mask`;
    `human_agent` blocks in a pygame event loop waiting for a valid mouse
    click on the board.
  - `close()` calls `pygame.quit()`.

## Environment / dependencies

- Requires Python `>=3.8` (per `setup.py`), developed/tested here under
  Python 3.11.
- Runtime dependencies used by the code: `gymnasium`, `numpy`, `pygame`.
  **None of these are declared in `setup.py`'s `install_requires`** — the
  README instead tells users to `pip install pygame` (and `ray[rllib]`)
  manually before installing this package via
  `pip install git+https://github.com/mo-kazuya/gym_othello.git`. If you
  add real dependency management, update `setup.py` and the README
  together rather than only one.
- `example/train_dqn.py`, `example/train_wthor.py`, and
  `example/train_ppo_transformer.py` additionally require `torch`
  (PyTorch); the package itself does not depend on it — keep RL-framework
  dependencies confined to `example/`.
- `example/train_wthor.py` documents the WTHOR (.wtb) binary format in its
  header comment, including the color mapping caveat: WTHOR/standard
  Othello has black moving first, while this env has white moving first
  with swapped initial colors — the geometry is identical, so WTHOR games
  replay directly through `Board` with "WTHOR black ≡ env white".
- `pygame` (and its font subsystem) is only touched when
  `render_mode="human"`; headless/training usage (`render_mode=None`)
  never imports/inits pygame's display or font machinery beyond what's
  already loaded at module import time.
- `atari.ttf` must stay listed in `MANIFEST.in` and referenced via
  `os.path.join(os.path.dirname(__file__), 'atari.ttf')` — both
  `Board.__init__` and `OthelloEnv.reset()` load the font this way now (the
  latter previously used a hardcoded local path,
  `/work/misc/othello2/gym_othello2/atari.ttf`, which raised
  `FileNotFoundError` for anyone other than the original author on
  `render_mode="human"`; fixed).

## Development workflow

- No CI, linter, or formatter config exists in this repo — there is
  nothing else to run beyond the test suite below.
- **Tests**: `test/test_a.py` uses `unittest`. Run with:
  ```
  python -m unittest discover -s test
  ```
  or directly:
  ```
  python -m unittest test.test_a
  ```
  (Requires `gymnasium`, `numpy`, and `pygame` installed first, even though
  the tests don't use `render_mode="human"`, since the module imports
  `pygame` unconditionally at the top of `othello_env.py`.)
- **Editable install for local development**:
  ```
  pip install pygame gymnasium numpy
  pip install -e .
  ```
- There is no `requirements.txt`; dependencies must be installed manually
  per the README.

## Conventions to follow

- Code comments and commit messages in this repo are written in Japanese;
  match that style for consistency when editing existing comments, but
  don't force-translate unrelated code.
- Keep all game logic inside `Board`/`OthelloEnv` in
  `gym_othello/envs/othello_env.py` — the package is intentionally a
  single-file implementation; don't split it into multiple modules unless
  asked.
- Board coordinates: functions consistently take `(col, row)` as arguments
  but store/index as `self.board[row][col]`. Preserve this ordering when
  adding new methods to avoid subtle row/col swap bugs.
- `current_player` is tracked as an int (`PLAYER_WHITE`/`PLAYER_BLACK`) on
  `Board`, but as a string (`"white"`/`"black"`) on `OthelloEnv` — use
  `_get_player()`/`_get_opponent()` helpers on `OthelloEnv` rather than
  comparing directly against the int constants.
