# 深層学習を使わない、位置評価テーブル + ミニマックス探索によるオセロAIサンプル
#
# オセロの定石知識 ―― 「角を取ると有利」「角に隣接するマス(X打ち)は
# 相手に角を取らせやすく危険」「辺は比較的安全」―― を位置評価テーブルに
# 埋め込み、ミニマックス法 (alpha-beta枝刈り) で数手先まで読んで着手を選ぶ。
# ニューラルネットは一切使わない、古典的なゲーム木探索エージェント。
#
# 使い方:
#   python example/heuristic_agent.py                  # ランダムエージェントと対戦評価
#   python example/heuristic_agent.py --depth 5         # 探索深さを変更 (強くなるが遅くなる)
#   python example/heuristic_agent.py --opponent greedy # 探索なし・位置評価のみのエージェントと対戦
#   python example/heuristic_agent.py --opponent dqn    # 学習済みDQN (train_dqn.py) と対戦
#   python example/heuristic_agent.py --opponent ppo    # 学習済みPPO/Transformer と対戦

import argparse
import copy
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gym_othello.envs import OthelloEnv
from gym_othello.envs.othello_env import PLAYER_WHITE, PLAYER_BLACK

# オセロの古典的な位置評価テーブル。
# 角(120)は最優先。角に隣接するマス(X打ち・C打ち, -40/-20)は
# 早い段階で打つと相手に角を献上しやすいため大きな負の評価。
# 辺(20)は比較的安定、中央付近は小さめの正の評価とする。
POSITIONAL_WEIGHTS = np.array([
    [120, -20,  20,   5,   5,  20, -20, 120],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [120, -20,  20,   5,   5,  20, -20, 120],
])


def opponent_of(player):
    return PLAYER_WHITE if player == PLAYER_BLACK else PLAYER_BLACK


def evaluate_board(board, player):
    """player視点の静的評価値。位置評価テーブルの差に、着手可能数(mobility)
    の差 (相手の選択肢を減らすほど有利、というオセロの基本戦略) を加味する。"""
    arr = np.array(board.get_board())
    opp = opponent_of(player)

    positional = (np.sum(POSITIONAL_WEIGHTS[arr == player])
                  - np.sum(POSITIONAL_WEIGHTS[arr == opp]))

    my_moves = len(board.get_valid_moves(player))
    opp_moves = len(board.get_valid_moves(opp))
    if my_moves + opp_moves > 0:
        mobility = 100.0 * (my_moves - opp_moves) / (my_moves + opp_moves)
    else:
        mobility = 0.0

    return positional + 2.0 * mobility


def minimax(board, player, root_player, depth, alpha, beta,
            eval_fn=None, move_order_fn=None):
    """alpha-beta枝刈り付きミニマックス。root_player視点のスコアを返す。
    eval_fn(board, root_player) -> float を差し替えれば、末端局面の評価を
    位置評価テーブル以外 (例: 学習済みネットワークの価値ヘッド) に置き換えられる。
    move_order_fn(board, player, moves) -> moves は展開順を変えて枝刈り効率を
    上げるためのフック (例: 方策ネットワークのlogits順)。"""
    eval_fn = eval_fn or evaluate_board
    moves = board.get_valid_moves(player)
    opp = opponent_of(player)

    if not moves:
        if board.is_game_over():
            return eval_fn(board, root_player)
        # パス: 手番だけ交代して同じ深さで相手を読む
        return minimax(board, opp, root_player, depth - 1, alpha, beta,
                       eval_fn, move_order_fn)

    if depth == 0:
        return eval_fn(board, root_player)

    if move_order_fn:
        moves = move_order_fn(board, player, moves)

    maximizing = (player == root_player)
    best = -float('inf') if maximizing else float('inf')
    for col, row in moves:
        child = copy.deepcopy(board)
        child.place_piece(col, row, player)
        score = minimax(child, child.current_player, root_player,
                        depth - 1, alpha, beta, eval_fn, move_order_fn)
        if maximizing:
            best = max(best, score)
            alpha = max(alpha, best)
        else:
            best = min(best, score)
            beta = min(beta, best)
        if beta <= alpha:
            break  # 枝刈り
    return best


def minimax_action(board, player, depth=3, eval_fn=None, move_order_fn=None):
    """ミニマックス探索で最善手を選び、action (0-63) を返す。"""
    eval_fn = eval_fn or evaluate_board
    moves = board.get_valid_moves(player)
    if move_order_fn:
        moves = move_order_fn(board, player, moves)
    best_score, best_move = -float('inf'), moves[0]
    for col, row in moves:
        child = copy.deepcopy(board)
        child.place_piece(col, row, player)
        score = minimax(child, child.current_player, player,
                        depth - 1, -float('inf'), float('inf'),
                        eval_fn, move_order_fn)
        if score > best_score:
            best_score, best_move = score, (col, row)
    col, row = best_move
    return row * 8 + col


def greedy_action(board, player):
    """探索なし。位置評価テーブルが最大の手を選ぶだけの単純な比較対象
    (角が見えていれば必ず角を選ぶが、数手先は読まない)。"""
    moves = board.get_valid_moves(player)
    col, row = max(moves, key=lambda m: POSITIONAL_WEIGHTS[m[1], m[0]])
    return row * 8 + col


def board_player(env):
    return PLAYER_WHITE if env.current_player == 'white' else PLAYER_BLACK


def make_opponent(name):
    """(name, obs) -> action の対戦相手コールバックを返す。env を介して盤面を参照する。"""
    if name == 'random':
        return lambda env, obs: OthelloEnv.random_agent(obs)
    if name == 'greedy':
        return lambda env, obs: greedy_action(env.Board, board_player(env))
    if name == 'dqn':
        from train_dqn import QNet, select_action
        path = os.path.join(os.path.dirname(__file__), 'dqn_othello.pt')
        import torch
        net = QNet()
        net.load_state_dict(torch.load(path, map_location='cpu'))
        net.eval()
        return lambda env, obs: select_action(net, obs, epsilon=0.0)
    if name == 'ppo':
        from train_ppo_transformer import TransformerPolicy, greedy_action as ppo_greedy
        path = os.path.join(os.path.dirname(__file__), 'ppo_transformer_othello.pt')
        import torch
        net = TransformerPolicy()
        net.load_state_dict(torch.load(path, map_location='cpu'))
        net.eval()
        return lambda env, obs: ppo_greedy(net, obs)
    raise ValueError(f'unknown opponent: {name}')


def evaluate(depth, opponent_name, games, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    opponent = make_opponent(opponent_name)

    results = {'win': 0, 'lose': 0, 'draw': 0}
    env = OthelloEnv()
    total_time, total_moves = 0.0, 0

    for i in range(games):
        my_player = 'white' if i % 2 == 0 else 'black'
        obs, _ = env.reset()
        done = False
        while not done:
            player = env.current_player
            if player == my_player:
                t0 = time.perf_counter()
                action = minimax_action(env.Board, board_player(env), depth)
                total_time += time.perf_counter() - t0
                total_moves += 1
            else:
                action = opponent(env, obs[player])
            obs, _, terminateds, _, _ = env.step({player: action})
            done = terminateds['__all__']
        diff = env.Board.count_white() - env.Board.count_black()
        if my_player == 'black':
            diff = -diff
        if diff > 0:
            results['win'] += 1
        elif diff < 0:
            results['lose'] += 1
        else:
            results['draw'] += 1

    total = sum(results.values())
    avg_ms = 1000 * total_time / max(1, total_moves)
    print(f"minimax(depth={depth}) vs {opponent_name} ({total} games): "
          f"win {results['win']} / lose {results['lose']} / draw {results['draw']} "
          f"(win rate {results['win']/total:.1%}, avg {avg_ms:.1f} ms/move)")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Minimax + positional-evaluation Othello agent (no neural network)')
    parser.add_argument('--depth', type=int, default=3,
                        help='ミニマックス探索の深さ (大きいほど強いが遅い)')
    parser.add_argument('--opponent', choices=['random', 'greedy', 'dqn', 'ppo'],
                        default='random')
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    evaluate(args.depth, args.opponent, args.games, seed=args.seed)


if __name__ == '__main__':
    main()
