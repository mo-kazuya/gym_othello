# DQN による OthelloEnv の深層強化学習サンプル
#
# 白番・黒番で共有する1つのQネットワークを自己対戦で学習する。
# 観測が「自分の盤面・相手の盤面」の視点相対な2チャネルなので、
# 同じネットワークを両プレイヤーにそのまま使える。
#
# 使い方:
#   python example/train_dqn.py                # 学習して example/dqn_othello.pt に保存
#   python example/train_dqn.py --episodes 500 # エピソード数を指定
#   python example/train_dqn.py --eval-only    # 保存済みモデルをランダムエージェントと対戦評価

import argparse
import collections
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gym_othello.envs import OthelloEnv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dqn_othello.pt')


# Qネットワーク: (2,8,8) の盤面観測 -> 64マスそれぞれのQ値
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

    def forward(self, x):
        return self.net(x)


Transition = collections.namedtuple(
    'Transition', ['obs', 'action', 'reward', 'next_obs', 'next_mask', 'done'])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        obs = torch.as_tensor(
            np.stack([t.obs for t in batch]), dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor(
            [t.action for t in batch], dtype=torch.int64, device=DEVICE)
        rewards = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=DEVICE)
        next_obs = torch.as_tensor(
            np.stack([t.next_obs for t in batch]), dtype=torch.float32, device=DEVICE)
        next_masks = torch.as_tensor(
            np.stack([t.next_mask for t in batch]), dtype=torch.float32, device=DEVICE)
        dones = torch.as_tensor(
            [t.done for t in batch], dtype=torch.float32, device=DEVICE)
        return obs, actions, rewards, next_obs, next_masks, dones

    def __len__(self):
        return len(self.buf)


# action_mask で無効手を弾いた上で greedy / epsilon-greedy に手を選ぶ
def select_action(qnet, obs, epsilon):
    mask = obs['action_mask'].flatten()  # (64,) 1=有効手
    valid = np.flatnonzero(mask)
    if random.random() < epsilon:
        return int(np.random.choice(valid))
    with torch.no_grad():
        x = torch.as_tensor(
            obs['observation'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        q = qnet(x).squeeze(0).cpu().numpy()
    q[mask == 0] = -np.inf
    return int(np.argmax(q))


# 1エピソード自己対戦し、プレイヤーごとの遷移をバッファへ積む
def play_episode(env, qnet, buffer, epsilon):
    obs, _ = env.reset()
    # 各プレイヤーの「直前の自分の手」を保持し、次に手番が回ってきた時に遷移を確定する
    pending = {}       # player -> (observation, action)
    reward_acc = {'white': 0.0, 'black': 0.0}
    done = False

    while not done:
        player = env.current_player
        my_obs = obs[player]

        # 前回自分が打った手の遷移を確定（次状態=今の自分の観測）
        if player in pending:
            prev_obs, prev_act = pending[player]
            buffer.push(prev_obs, prev_act, reward_acc[player],
                        my_obs['observation'], my_obs['action_mask'].flatten(), 0.0)
            reward_acc[player] = 0.0

        action = select_action(qnet, my_obs, epsilon)
        pending[player] = (my_obs['observation'], action)

        obs, rewards, terminateds, _, _ = env.step({player: action})
        for p, r in rewards.items():
            reward_acc[p] += r
        done = terminateds['__all__']

    # 終局: 両プレイヤーの残り遷移を done=1 で確定（次状態は使われないのでダミー）
    dummy_obs = np.zeros((2, 8, 8), dtype=np.int8)
    dummy_mask = np.zeros(64, dtype=np.int8)
    for player, (prev_obs, prev_act) in pending.items():
        buffer.push(prev_obs, prev_act, reward_acc[player],
                    dummy_obs, dummy_mask, 1.0)


def train_step(qnet, target_net, buffer, optimizer, batch_size, gamma):
    obs, actions, rewards, next_obs, next_masks, dones = buffer.sample(batch_size)

    q = qnet(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_obs)
        # 無効手を -inf にして有効手の最大Q値を取る（全て無効=終局はdoneで消える）
        next_q[next_masks == 0] = -1e9
        max_next_q = next_q.max(dim=1).values
    target = rewards + gamma * (1.0 - dones) * max_next_q

    loss = nn.functional.smooth_l1_loss(q, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# 学習済みポリシー(greedy) vs ランダムエージェントで対戦評価
def evaluate(qnet, games=100):
    results = {'win': 0, 'lose': 0, 'draw': 0}
    env = OthelloEnv()
    for i in range(games):
        dqn_player = 'white' if i % 2 == 0 else 'black'  # 手番を交互に持つ
        obs, _ = env.reset()
        done = False
        while not done:
            player = env.current_player
            if player == dqn_player:
                action = select_action(qnet, obs[player], epsilon=0.0)
            else:
                action = OthelloEnv.random_agent(obs[player])
            obs, _, terminateds, _, _ = env.step({player: action})
            done = terminateds['__all__']
        diff = env.Board.count_white() - env.Board.count_black()
        if dqn_player == 'black':
            diff = -diff
        if diff > 0:
            results['win'] += 1
        elif diff < 0:
            results['lose'] += 1
        else:
            results['draw'] += 1
    return results


def main():
    parser = argparse.ArgumentParser(description='DQN self-play training for OthelloEnv')
    parser.add_argument('--episodes', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--buffer-size', type=int, default=50000)
    parser.add_argument('--eps-start', type=float, default=1.0)
    parser.add_argument('--eps-end', type=float, default=0.05)
    parser.add_argument('--eps-decay-episodes', type=int, default=300)
    parser.add_argument('--target-update', type=int, default=500, help='target net 同期間隔(ステップ)')
    parser.add_argument('--eval-every', type=int, default=50, help='評価間隔(エピソード)')
    parser.add_argument('--eval-games', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    qnet = QNet().to(DEVICE)

    if args.eval_only:
        qnet.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        results = evaluate(qnet, games=args.eval_games)
        total = sum(results.values())
        print(f"vs random ({total} games): "
              f"win {results['win']} / lose {results['lose']} / draw {results['draw']} "
              f"(win rate {results['win']/total:.1%})")
        return

    target_net = QNet().to(DEVICE)
    target_net.load_state_dict(qnet.state_dict())
    optimizer = torch.optim.Adam(qnet.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)
    env = OthelloEnv()

    global_step = 0
    for episode in range(1, args.episodes + 1):
        # epsilon を線形に減衰させる
        frac = min(1.0, episode / args.eps_decay_episodes)
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)

        play_episode(env, qnet, buffer, epsilon)

        # 1エピソードごとにまとめて学習(おおよそ1手=1更新)
        if len(buffer) >= args.batch_size:
            for _ in range(60):
                train_step(qnet, target_net, buffer, optimizer,
                           args.batch_size, args.gamma)
                global_step += 1
                if global_step % args.target_update == 0:
                    target_net.load_state_dict(qnet.state_dict())

        if episode % args.eval_every == 0:
            results = evaluate(qnet, games=args.eval_games)
            total = sum(results.values())
            print(f"episode {episode:4d}  epsilon={epsilon:.3f}  "
                  f"vs random: win {results['win']:3d} / lose {results['lose']:3d} "
                  f"/ draw {results['draw']:2d}  (win rate {results['win']/total:.1%})")

    torch.save(qnet.state_dict(), MODEL_PATH)
    print(f"saved model to {MODEL_PATH}")


if __name__ == '__main__':
    main()
