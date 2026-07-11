# Transformerの価値ヘッドをミニマックス探索の末端評価に使うハイブリッドエージェント
# (AlphaZero的な「ネットワーク + 探索」構成)
#
# なぜ必要か:
#   train_ppo_transformer.py の Transformer は1回の順伝播だけで着手を決める、
#   先読みをしないポリシーである。一方 heuristic_agent.py のミニマックスは
#   毎手、実際に数手先の盤面をシミュレートして評価する。この「探索の有無」
#   という構造的な差のせいで、自己対戦だけで学習したTransformerは
#   ミニマックス(depth=3)に一度も勝てなかった (train_ppo_transformer.py に
#   ミニマックス対戦を混ぜて再学習しても、この差は埋まらなかった)。
#
#   そこで探索そのものは heuristic_agent.py のミニマックス(alpha-beta)に
#   任せたまま、末端局面の評価だけを「角=120, X打ち=-40...」の手作り
#   位置評価テーブルではなく、学習済みTransformerの価値ヘッドに置き換える。
#   さらに方策ヘッドのlogitsで着手の展開順を並べ替え、alpha-beta の
#   枝刈り効率を上げる (有望な手を先に読むほど早く枝刈りできる)。
#
# 使い方:
#   python example/transformer_minimax_agent.py --depth 3                      # 位置評価テーブル版ミニマックス(同じ深さ)と対戦
#   python example/transformer_minimax_agent.py --depth 2 --opponent-depth 3   # 探索深さが違う相手とも対戦可能
#   python example/transformer_minimax_agent.py --depth 3 --opponent random

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gym_othello.envs import OthelloEnv
from heuristic_agent import (
    minimax_action, opponent_of, board_player, make_opponent as make_table_opponent,
)
from train_ppo_transformer import TransformerPolicy, MODEL_PATH as PPO_MODEL_PATH

DEVICE = torch.device('cpu')  # 探索中に小さな推論を多数行うため、GPU転送コストを避けCPUで統一


def board_to_obs(board, player):
    """視点相対の (2,8,8) 観測 [自分の石, 相手の石] を作る (OthelloEnv._get_obs と同形式)。"""
    arr = np.array(board.get_board())
    opp = opponent_of(player)
    mine = np.int8(arr == player)
    theirs = np.int8(arr == opp)
    return np.stack((mine, theirs))


def make_nn_eval_fn(net):
    """末端局面を Transformer の価値ヘッドで評価する eval_fn(board, root_player) を返す。"""
    @torch.no_grad()
    def _eval(board, root_player):
        x = torch.as_tensor(board_to_obs(board, root_player), dtype=torch.float32,
                            device=DEVICE).unsqueeze(0)
        _, value = net(x)
        return value.item()
    return _eval


def make_nn_move_order_fn(net):
    """方策ヘッドのlogitsが高い順に手を並べ替える move_order_fn(board, player, moves) を返す。
    有望な手を先に探索するほど alpha-beta の枝刈りが早く効く。"""
    @torch.no_grad()
    def _order(board, player, moves):
        x = torch.as_tensor(board_to_obs(board, player), dtype=torch.float32,
                            device=DEVICE).unsqueeze(0)
        logits, _ = net(x)
        logits = logits.squeeze(0).cpu().numpy()
        return sorted(moves, key=lambda m: -logits[m[1] * 8 + m[0]])
    return _order


def load_policy(path=PPO_MODEL_PATH):
    net = TransformerPolicy().to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net


def nn_guided_action(board, player, net, depth, eval_fn, order_fn):
    return minimax_action(board, player, depth=depth, eval_fn=eval_fn, move_order_fn=order_fn)


def evaluate(depth, opponent_name, opponent_depth, games, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    net = load_policy()
    eval_fn = make_nn_eval_fn(net)
    order_fn = make_nn_move_order_fn(net)

    if opponent_name == 'table-minimax':
        from heuristic_agent import minimax_action as table_minimax_action
        opponent = lambda env, obs: table_minimax_action(
            env.Board, board_player(env), depth=opponent_depth)
        opp_label = f'table-minimax(depth={opponent_depth})'
    else:
        opponent = make_table_opponent(opponent_name)
        opp_label = opponent_name

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
                action = nn_guided_action(env.Board, board_player(env), net, depth,
                                          eval_fn, order_fn)
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
    print(f"nn-guided-minimax(depth={depth}) vs {opp_label} ({total} games): "
          f"win {results['win']} / lose {results['lose']} / draw {results['draw']} "
          f"(win rate {results['win']/total:.1%}, avg {avg_ms:.1f} ms/move)")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Minimax search with a Transformer value head as the leaf evaluator '
                    '(policy head used for move ordering)')
    parser.add_argument('--depth', type=int, default=3,
                        help='ミニマックス探索の深さ')
    parser.add_argument('--opponent', choices=['table-minimax', 'random', 'greedy', 'dqn', 'ppo'],
                        default='table-minimax')
    parser.add_argument('--opponent-depth', type=int, default=3,
                        help='--opponent table-minimax のときの相手側の探索深さ')
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    evaluate(args.depth, args.opponent, args.opponent_depth, args.games, seed=args.seed)


if __name__ == '__main__':
    main()
