# AlphaZero方式の蒸留 (expert iteration) で Transformer を再学習するサンプル
#
# transformer_minimax_agent.py の実験で、「探索を足しても、末端評価
# (価値ヘッド) の精度が手作りの位置評価テーブルに及ばなければ勝てない」
# ことが分かった。そこで AlphaZero と同じ発想で、
#
#   1. 現在のネットワークを末端評価に使ったミニマックス探索で自己対戦する
#      (探索がネットワーク単体より強い手を指す = 教師の増幅)
#   2. 「探索が選んだ手」を方策ヘッドの教師、「終局の石差」を価値ヘッドの
#      教師として、ネットワークを教師あり学習で更新する (蒸留)
#   3. 強くなったネットワークで 1. に戻る
#
# というループを回す。探索は heuristic_agent.py のミニマックス(alpha-beta)、
# 局面の多様化には env の random opening offset (reset options={'offset': N})
# を使い、ランダムに進めた序盤以降の「探索した手」だけを教師データに記録する。
#
# 価値ヘッドの教師 z は終局石差を [-1,1] に正規化したもの (diff/64)。
# PPO で学習した価値ヘッドは報酬スケール (±13程度) を予測しているため、
# 蒸留の初回でスケールが合わないが、学習が進めば適応する。
#
# 使い方:
#   python example/train_alphazero_distill.py                # 既存PPOモデルを初期値に蒸留ループ
#   python example/train_alphazero_distill.py --iterations 8 --games-per-iter 80
#   python example/train_alphazero_distill.py --eval-only    # 保存済み蒸留モデルで最終評価
#     (nn-guided-minimax vs table-minimax をランダム開局で対戦。決定論的な
#      2パターン対局によるノイズを避けるため、評価は常にランダム開局で行う)

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gym_othello.envs import OthelloEnv
from heuristic_agent import (
    minimax_action as table_or_nn_minimax_action, board_player,
)
from train_ppo_transformer import TransformerPolicy, MODEL_PATH as PPO_MODEL_PATH
from transformer_minimax_agent import (
    board_to_obs, make_nn_eval_fn, make_nn_move_order_fn,
)

DEVICE = torch.device('cpu')  # 探索中の多数の小さな推論はCPUの方が転送コストがなく速い
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          'alphazero_transformer_othello.pt')


# ---------------------------------------------------------------------------
# 自己対戦データ生成 (探索 = 教師)
# ---------------------------------------------------------------------------

def selfplay_game(env, eval_fn, order_fn, search_depth, max_offset):
    """ランダム開局から NN誘導ミニマックスの自己対戦を1局行い、
    (観測, 探索が選んだ手, 手番プレイヤー) と終局石差を返す。
    ランダム開局部分は探索していないので教師データに含めない。"""
    offset = random.randint(0, max_offset)
    obs, _ = env.reset(options={'offset': offset})
    records = []  # (obs(2,8,8), action, player_str)
    done = env.Board.is_game_over()

    while not done:
        player = env.current_player
        my_obs = obs[player]
        action = table_or_nn_minimax_action(
            env.Board, board_player(env), depth=search_depth,
            eval_fn=eval_fn, move_order_fn=order_fn)
        records.append((my_obs['observation'], action, player))
        obs, _, terminateds, _, _ = env.step({player: action})
        done = terminateds['__all__']

    diff = env.Board.count_white() - env.Board.count_black()
    return records, diff


def generate_dataset(env, net, n_games, search_depth, max_offset):
    """n_games 分の自己対戦から (観測, 探索手, 石差z) の教師データを作る。"""
    eval_fn = make_nn_eval_fn(net)
    order_fn = make_nn_move_order_fn(net)
    xs, ys, zs = [], [], []
    for _ in range(n_games):
        records, diff = selfplay_game(env, eval_fn, order_fn,
                                      search_depth, max_offset)
        for obs_arr, action, player in records:
            # 価値の教師: その手番プレイヤー視点の終局石差を [-1,1] に正規化
            z = diff / 64.0 if player == 'white' else -diff / 64.0
            xs.append(obs_arr)
            ys.append(action)
            zs.append(z)
    return (torch.as_tensor(np.stack(xs), dtype=torch.float32),
            torch.as_tensor(ys, dtype=torch.int64),
            torch.as_tensor(zs, dtype=torch.float32))


# ---------------------------------------------------------------------------
# 蒸留 (教師あり学習)
# ---------------------------------------------------------------------------

def distill(net, optimizer, dataset, epochs, batch_size, value_coef):
    xs, ys, zs = dataset
    n = len(xs)
    stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'n': 0}
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits, value = net(xs[idx].to(DEVICE))
            policy_loss = nn.functional.cross_entropy(logits, ys[idx].to(DEVICE))
            value_loss = nn.functional.mse_loss(value, zs[idx].to(DEVICE))
            loss = policy_loss + value_coef * value_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            stats['policy_loss'] += policy_loss.item()
            stats['value_loss'] += value_loss.item()
            stats['n'] += 1
    net.eval()
    return {k: v / stats['n'] for k, v in stats.items() if k != 'n'}


# ---------------------------------------------------------------------------
# 評価: ランダム開局で nn-guided-minimax vs table-minimax
# ---------------------------------------------------------------------------

def evaluate_vs_table(net, my_depth, table_depth, games, max_offset=6):
    """ランダム開局 (決定論的2パターン対局のノイズ回避) で
    NN誘導ミニマックス vs 位置評価テーブル版ミニマックスを対戦させる。"""
    eval_fn = make_nn_eval_fn(net)
    order_fn = make_nn_move_order_fn(net)
    results = {'win': 0, 'lose': 0, 'draw': 0}
    env = OthelloEnv()
    for i in range(games):
        my_player = 'white' if i % 2 == 0 else 'black'
        obs, _ = env.reset(options={'offset': random.randint(2, max_offset)})
        done = env.Board.is_game_over()
        while not done:
            player = env.current_player
            if player == my_player:
                action = table_or_nn_minimax_action(
                    env.Board, board_player(env), depth=my_depth,
                    eval_fn=eval_fn, move_order_fn=order_fn)
            else:
                action = table_or_nn_minimax_action(
                    env.Board, board_player(env), depth=table_depth)
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
    return results


def report(label, results):
    total = sum(results.values())
    print(f"{label} ({total} games): win {results['win']} / lose {results['lose']} "
          f"/ draw {results['draw']}  (win rate {results['win']/total:.1%})",
          flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='AlphaZero-style distillation: retrain the Transformer from '
                    'search-guided self-play')
    parser.add_argument('--iterations', type=int, default=6)
    parser.add_argument('--games-per-iter', type=int, default=60)
    parser.add_argument('--search-depth', type=int, default=2,
                        help='自己対戦データ生成時の探索深さ')
    parser.add_argument('--max-offset', type=int, default=8,
                        help='ランダム開局の最大手数 (局面多様化)')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--value-coef', type=float, default=1.0)
    parser.add_argument('--init-from', default=PPO_MODEL_PATH,
                        help='初期重み (デフォルト: PPO学習済みモデル)')
    parser.add_argument('--eval-games', type=int, default=20)
    parser.add_argument('--eval-depth', type=int, default=2,
                        help='反復ごとの中間評価の探索深さ (双方)')
    parser.add_argument('--final-eval-depth', type=int, default=3,
                        help='最終評価の探索深さ (双方)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    net = TransformerPolicy().to(DEVICE)

    if args.eval_only:
        net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        net.eval()
        report(f'nn-guided(depth={args.final_eval_depth}) vs table-minimax(depth={args.final_eval_depth})',
               evaluate_vs_table(net, args.final_eval_depth,
                                 args.final_eval_depth, args.eval_games))
        return

    net.load_state_dict(torch.load(args.init_from, map_location=DEVICE))
    net.eval()
    print(f'initialized weights from {args.init_from}', flush=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    env = OthelloEnv()

    # 蒸留前のベースライン (中間評価と同条件)
    report(f'baseline nn-guided(depth={args.eval_depth}) vs table-minimax(depth={args.eval_depth})',
           evaluate_vs_table(net, args.eval_depth, args.eval_depth, args.eval_games))

    for it in range(1, args.iterations + 1):
        t0 = time.perf_counter()
        dataset = generate_dataset(env, net, args.games_per_iter,
                                   args.search_depth, args.max_offset)
        gen_sec = time.perf_counter() - t0

        stats = distill(net, optimizer, dataset, args.epochs,
                        args.batch_size, args.value_coef)
        print(f"iter {it}: {len(dataset[0])} positions "
              f"(gen {gen_sec:.0f}s)  policy_loss {stats['policy_loss']:.4f}  "
              f"value_loss {stats['value_loss']:.4f}", flush=True)

        r = evaluate_vs_table(net, args.eval_depth, args.eval_depth,
                              args.eval_games)
        report(f'  iter {it} nn-guided(depth={args.eval_depth}) '
               f'vs table-minimax(depth={args.eval_depth})', r)

        torch.save(net.state_dict(), MODEL_PATH)

    print(f'saved model to {MODEL_PATH}', flush=True)

    # 最終評価: 本来の目標である depth=3 同士
    report(f'FINAL nn-guided(depth={args.final_eval_depth}) '
           f'vs table-minimax(depth={args.final_eval_depth})',
           evaluate_vs_table(net, args.final_eval_depth,
                             args.final_eval_depth, args.eval_games))


if __name__ == '__main__':
    main()
