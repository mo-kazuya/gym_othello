# Transformer 方策を PPO で強化学習するサンプル
#
# 盤面の64マスをトークン列とみなし、Transformer エンコーダで方策と
# 状態価値を出力するネットワークを、自己対戦 + PPO (clipped surrogate)
# で学習する。白番・黒番は視点相対の観測を共有する1つのネットワーク。
#
# 使い方:
#   python example/train_ppo_transformer.py                  # 学習して example/ppo_transformer_othello.pt に保存
#   python example/train_ppo_transformer.py --iterations 200 # 反復回数を指定
#   python example/train_ppo_transformer.py --eval-only      # 保存済みモデルをランダムエージェントと対戦評価

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gym_othello.envs import OthelloEnv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ppo_transformer_othello.pt')


# Transformer 方策・価値ネットワーク
#   入力: (B,2,8,8) の視点相対観測 [自分の石, 相手の石]
#   各マスを「空き=0 / 自分=1 / 相手=2」の3種トークンとして埋め込み、
#   学習可能な位置埋め込みを加えて Transformer エンコーダに通す。
#   方策: 各トークンから1次元 -> 64マスのlogits
#   価値: 全トークンの平均プーリング -> スカラー
class TransformerPolicy(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.token_emb = nn.Embedding(3, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 64, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.policy_head = nn.Linear(d_model, 1)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        # (B,2,8,8) -> トークンID (B,64): 空き=0, 自分=1, 相手=2
        tokens = (x[:, 0] + 2 * x[:, 1]).reshape(x.size(0), 64).long()
        h = self.token_emb(tokens) + self.pos_emb
        h = self.encoder(h)
        logits = self.policy_head(h).squeeze(-1)      # (B,64)
        value = self.value_head(h.mean(dim=1)).squeeze(-1)  # (B,)
        return logits, value


def masked_dist(logits, mask):
    """無効手のlogitsを潰した Categorical 分布を返す。mask: (B,64) 1=有効手"""
    logits = logits.masked_fill(mask == 0, -1e9)
    return Categorical(logits=logits)


@torch.no_grad()
def sample_action(net, obs):
    """対戦データ収集用: 行動・log確率・状態価値を返す。"""
    x = torch.as_tensor(obs['observation'], dtype=torch.float32,
                        device=DEVICE).unsqueeze(0)
    mask = torch.as_tensor(obs['action_mask'].flatten(), dtype=torch.float32,
                           device=DEVICE).unsqueeze(0)
    logits, value = net(x)
    dist = masked_dist(logits, mask)
    action = dist.sample()
    return int(action.item()), dist.log_prob(action).item(), value.item()


@torch.no_grad()
def greedy_action(net, obs):
    """評価用: 最尤手を返す。"""
    x = torch.as_tensor(obs['observation'], dtype=torch.float32,
                        device=DEVICE).unsqueeze(0)
    logits, _ = net(x)
    logits = logits.squeeze(0).cpu().numpy()
    logits[obs['action_mask'].flatten() == 0] = -np.inf
    return int(np.argmax(logits))


# 1エピソード自己対戦し、プレイヤーごとの軌跡を返す。
# 交互手番のため「自分の着手 -> 次の自分の手番」を1遷移とし、
# その間に発生した報酬 (4隅ボーナス・終局報酬) を自分の遷移に集約する。
def play_episode(env, net):
    obs, _ = env.reset()
    traj = {p: {'obs': [], 'mask': [], 'act': [], 'logp': [],
                'val': [], 'rew': []} for p in ('white', 'black')}
    reward_acc = {'white': 0.0, 'black': 0.0}
    done = False

    while not done:
        player = env.current_player
        my_obs = obs[player]
        t = traj[player]

        # 前回の自分の着手に対する報酬を確定
        if t['act']:
            t['rew'].append(reward_acc[player])
            reward_acc[player] = 0.0

        action, logp, value = sample_action(net, my_obs)
        t['obs'].append(my_obs['observation'])
        t['mask'].append(my_obs['action_mask'].flatten())
        t['act'].append(action)
        t['logp'].append(logp)
        t['val'].append(value)

        obs, rewards, terminateds, _, _ = env.step({player: action})
        for p, r in rewards.items():
            reward_acc[p] += r
        done = terminateds['__all__']

    # 終局: 最後の着手の報酬を確定
    for player in ('white', 'black'):
        if traj[player]['act']:
            traj[player]['rew'].append(reward_acc[player])
    return traj


def compute_gae(rewards, values, gamma, lam):
    """1プレイヤー分の軌跡 (終局で必ず終わる) の GAE と収益を計算する。"""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_value = values[t + 1] if t + 1 < n else 0.0  # 終局後の価値は0
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * last_gae
        adv[t] = last_gae
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


def collect_batch(env, net, n_games, gamma, lam):
    """n_games 分の自己対戦データを集めて学習用テンソルに整形する。"""
    buf = {k: [] for k in ('obs', 'mask', 'act', 'logp', 'adv', 'ret')}
    for _ in range(n_games):
        traj = play_episode(env, net)
        for player in ('white', 'black'):
            t = traj[player]
            if not t['act']:
                continue
            adv, ret = compute_gae(t['rew'], t['val'], gamma, lam)
            buf['obs'].extend(t['obs'])
            buf['mask'].extend(t['mask'])
            buf['act'].extend(t['act'])
            buf['logp'].extend(t['logp'])
            buf['adv'].extend(adv)
            buf['ret'].extend(ret)
    return (
        torch.as_tensor(np.stack(buf['obs']), dtype=torch.float32, device=DEVICE),
        torch.as_tensor(np.stack(buf['mask']), dtype=torch.float32, device=DEVICE),
        torch.as_tensor(buf['act'], dtype=torch.int64, device=DEVICE),
        torch.as_tensor(buf['logp'], dtype=torch.float32, device=DEVICE),
        torch.as_tensor(np.asarray(buf['adv']), dtype=torch.float32, device=DEVICE),
        torch.as_tensor(np.asarray(buf['ret']), dtype=torch.float32, device=DEVICE),
    )


def ppo_update(net, optimizer, batch, clip_eps, value_coef, entropy_coef,
               update_epochs, minibatch_size):
    obs, mask, act, old_logp, adv, ret = batch
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'n': 0}
    n = len(obs)
    for _ in range(update_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, minibatch_size):
            idx = perm[i:i + minibatch_size]
            logits, value = net(obs[idx])
            dist = masked_dist(logits, mask[idx])
            logp = dist.log_prob(act[idx])
            ratio = torch.exp(logp - old_logp[idx])

            surr1 = ratio * adv[idx]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(value, ret[idx])
            entropy = dist.entropy().mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            stats['policy_loss'] += policy_loss.item()
            stats['value_loss'] += value_loss.item()
            stats['entropy'] += entropy.item()
            stats['n'] += 1
    return {k: v / stats['n'] for k, v in stats.items() if k != 'n'}


def evaluate(net, games=100):
    """greedy方策 vs ランダムエージェント (手番は交互に持つ)。"""
    results = {'win': 0, 'lose': 0, 'draw': 0}
    env = OthelloEnv()
    for i in range(games):
        my_player = 'white' if i % 2 == 0 else 'black'
        obs, _ = env.reset()
        done = False
        while not done:
            player = env.current_player
            if player == my_player:
                action = greedy_action(net, obs[player])
            else:
                action = OthelloEnv.random_agent(obs[player])
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


def main():
    parser = argparse.ArgumentParser(
        description='PPO training of a Transformer policy for OthelloEnv')
    parser.add_argument('--iterations', type=int, default=150)
    parser.add_argument('--games-per-iter', type=int, default=16)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-eps', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--update-epochs', type=int, default=4)
    parser.add_argument('--minibatch-size', type=int, default=256)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--eval-games', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    net = TransformerPolicy(d_model=args.d_model, nhead=args.nhead,
                            num_layers=args.num_layers).to(DEVICE)

    if args.eval_only:
        net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        net.eval()
        results = evaluate(net, games=args.eval_games)
        total = sum(results.values())
        print(f"vs random ({total} games): "
              f"win {results['win']} / lose {results['lose']} / draw {results['draw']} "
              f"(win rate {results['win']/total:.1%})")
        return

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    env = OthelloEnv()

    for it in range(1, args.iterations + 1):
        net.eval()  # 収集時はdropout等を無効化 (本モデルはdropout=0だが慣例として)
        batch = collect_batch(env, net, args.games_per_iter,
                              args.gamma, args.gae_lambda)
        net.train()
        stats = ppo_update(net, optimizer, batch, args.clip_eps,
                           args.value_coef, args.entropy_coef,
                           args.update_epochs, args.minibatch_size)

        if it % args.eval_every == 0:
            net.eval()
            results = evaluate(net, games=args.eval_games)
            total = sum(results.values())
            print(f"iter {it:4d}  policy_loss {stats['policy_loss']:+.4f}  "
                  f"value_loss {stats['value_loss']:.4f}  entropy {stats['entropy']:.3f}  "
                  f"vs random: win {results['win']:3d} / lose {results['lose']:3d} "
                  f"/ draw {results['draw']:2d}  (win rate {results['win']/total:.1%})")

    torch.save(net.state_dict(), MODEL_PATH)
    print(f'saved model to {MODEL_PATH}')


if __name__ == '__main__':
    main()
