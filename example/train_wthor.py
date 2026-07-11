# WTHOR 対局データベースを使った DNN (方策ネットワーク) の教師あり学習サンプル
#
# WTHOR はフランスオセロ連盟 (FFO) が配布している対局データベース形式。
# 各対局の着手列から「盤面 -> 打たれた手」のペアを作り、
# 打ち手を予測する方策ネットワークを cross-entropy で学習する。
#
# データの入手:
#   https://www.ffothello.org/informatique/la-base-wthor/ から
#   WTH_YYYY.ZIP をダウンロードし、中の .wtb ファイルを
#   example/wthor_data/ に置く。
#
# 使い方:
#   python example/train_wthor.py --make-synthetic 2000  # 実データが無い場合の動作確認用に
#                                                        # WTHOR形式の合成対局データを生成
#   python example/train_wthor.py                        # 学習して example/policy_wthor.pt に保存
#   python example/train_wthor.py --winner-only          # 勝者の手のみ学習
#   python example/train_wthor.py --eval-only            # 保存済みモデルを対戦評価
#
# WTHOR (.wtb) フォーマット:
#   ヘッダ 16 バイト:
#     [0]     作成日: 世紀   [1] 年   [2] 月   [3] 日
#     [4:8]   対局数 N (uint32 LE)
#     [8:10]  レコード数 (uint16 LE, .jou/.trn 用)
#     [10:12] 対局年 (uint16 LE)
#     [12]    盤サイズ (0 または 8 = 8x8, 10 = 10x10)
#     [13]    対局種別   [14] 理論スコアの探索深さ   [15] 予約
#   対局レコード 68 バイト x N:
#     [0:2] 大会ID  [2:4] 黒番プレイヤーID  [4:6] 白番プレイヤーID (各 uint16 LE)
#     [6]   黒の実スコア (空きマスは勝者に加算した黒石数)
#     [7]   理論スコア
#     [8:68] 着手 60 バイト。各バイトは 10*段 + 筋 (1始まり)。
#            例: f5 = 56 (筋f=6, 段5)。パスは記録されず、終局後は 0 詰め。
#
# 色の対応に注意:
#   標準オセロ (WTHOR) は黒が先手だが、この環境 (OthelloEnv/Board) は
#   白 (PLAYER_WHITE) が先手で初期配置の色も入れ替わっている。
#   つまり幾何は同一で「WTHORの黒 = 環境の白」という色の呼び替えだけなので、
#   着手列を Board でそのまま順に再生すれば正しく復元できる。

import argparse
import glob
import os
import random
import struct
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gym_othello.envs import OthelloEnv
from gym_othello.envs.othello_env import Board, PLAYER_WHITE, PLAYER_BLACK

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'wthor_data')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'policy_wthor.pt')
DQN_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dqn_othello.pt')


# 方策ネットワーク: (2,8,8) の視点相対観測 -> 64マスのlogits
# (train_dqn.py の QNet と同一構成。同じ入力形式なので相互比較できる)
class PolicyNet(nn.Module):
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


# ---------------------------------------------------------------------------
# WTHOR パーサ
# ---------------------------------------------------------------------------

def parse_wtb(path):
    """1つの .wtb ファイルから [(moves, black_score), ...] を返す。
    moves は着手バイト列 (0詰めを除去済み)。"""
    with open(path, 'rb') as f:
        data = f.read()
    if len(data) < 16:
        raise ValueError(f'{path}: too short for a WTHOR header')
    n_games = struct.unpack_from('<I', data, 4)[0]
    board_size = data[12]
    if board_size not in (0, 8):
        print(f'skip {os.path.basename(path)}: board size {board_size} (10x10 は非対応)')
        return []
    games = []
    off = 16
    for _ in range(n_games):
        rec = data[off:off + 68]
        off += 68
        if len(rec) < 68:
            break  # 壊れたファイル末尾
        black_score = rec[6]
        moves = [b for b in rec[8:68] if b != 0]
        games.append((moves, black_score))
    return games


def board_planes(board, player):
    """視点相対の (2,8,8) 観測 [自分の石, 相手の石] を作る (OthelloEnv._get_obs と同形式)。"""
    arr = np.array(board.get_board())
    mine = np.int8(arr == player)
    opp = np.int8(arr == (PLAYER_BLACK if player == PLAYER_WHITE else PLAYER_WHITE))
    return np.stack((mine, opp))


def game_to_samples(moves):
    """1対局の着手列を Board で再生し (観測, 着手, 手番) のリストへ変換する。
    不正な手を含む対局 (壊れたレコード) は None を返して呼び出し元で捨てる。"""
    board = Board()
    samples = []
    for b in moves:
        col = b % 10 - 1
        row = b // 10 - 1
        if not (0 <= col < 8 and 0 <= row < 8):
            return None
        player = board.current_player  # place_piece がパスを自動処理する
        if not board.is_valid_move(col, row, player):
            return None
        samples.append((board_planes(board, player), row * 8 + col, player))
        board.place_piece(col, row, player)
    return samples


def load_dataset(data_dir, winner_only=False, val_ratio=0.1, seed=0):
    """data_dir 内の全 .wtb を読み、対局単位で train/val に分割して tensor を返す。"""
    paths = sorted(glob.glob(os.path.join(data_dir, '*.wtb'))
                   + glob.glob(os.path.join(data_dir, '*.WTB')))
    if not paths:
        return None
    all_games = []
    for path in paths:
        games = parse_wtb(path)
        print(f'{os.path.basename(path)}: {len(games)} games')
        all_games.extend(games)

    rng = random.Random(seed)
    rng.shuffle(all_games)

    skipped = 0
    per_game_samples = []
    for moves, black_score in all_games:
        samples = game_to_samples(moves)
        if samples is None or len(samples) == 0:
            skipped += 1
            continue
        if winner_only:
            # black_score > 32 なら WTHOR の黒 (=環境の白, 先手) の勝ち
            if black_score > 32:
                winner = PLAYER_WHITE
            elif black_score < 32:
                winner = PLAYER_BLACK
            else:
                skipped += 1  # 引き分けは除外
                continue
            samples = [s for s in samples if s[2] == winner]
        per_game_samples.append(samples)
    if skipped:
        print(f'skipped {skipped} games (corrupt record or filtered out)')

    n_val_games = max(1, int(len(per_game_samples) * val_ratio))
    val_games = per_game_samples[:n_val_games]
    train_games = per_game_samples[n_val_games:]

    def to_tensors(games):
        flat = [s for g in games for s in g]
        x = torch.as_tensor(np.stack([s[0] for s in flat]), dtype=torch.float32)
        y = torch.as_tensor([s[1] for s in flat], dtype=torch.int64)
        return x, y

    train_x, train_y = to_tensors(train_games)
    val_x, val_y = to_tensors(val_games)
    print(f'dataset: {len(train_games)} train games ({len(train_x)} positions), '
          f'{len(val_games)} val games ({len(val_x)} positions)')
    return train_x, train_y, val_x, val_y


# ---------------------------------------------------------------------------
# 合成データ生成 (実データが手に入らない環境での動作確認用)
# ---------------------------------------------------------------------------

def make_synthetic(data_dir, n_games, epsilon=0.1, seed=0):
    """自己対戦の棋譜を WTHOR バイナリ形式で書き出す。
    学習済み DQN (dqn_othello.pt) があればそれを打ち手に使い、無ければランダム。"""
    from train_dqn import QNet, select_action

    rng = np.random.RandomState(seed)
    random.seed(seed)

    qnet = None
    if os.path.exists(DQN_MODEL_PATH):
        qnet = QNet().to(DEVICE)
        qnet.load_state_dict(torch.load(DQN_MODEL_PATH, map_location=DEVICE))
        qnet.eval()
        print(f'generating {n_games} games with DQN expert (epsilon={epsilon})')
    else:
        print(f'generating {n_games} games with random players')

    env = OthelloEnv()
    records = []
    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        move_bytes = []
        while not done:
            player = env.current_player
            if qnet is not None:
                action = select_action(qnet, obs[player], epsilon)
            else:
                action = OthelloEnv.random_agent(obs[player])
            col, row = action % 8, action // 8
            move_bytes.append((row + 1) * 10 + (col + 1))
            obs, _, terminateds, _, _ = env.step({player: action})
            done = terminateds['__all__']
        # 黒の実スコア: WTHORの黒=環境の白。空きマスは勝者に加算する慣例
        wc, bc = env.Board.count_white(), env.Board.count_black()
        empty = 64 - wc - bc
        score = wc + empty if wc > bc else wc
        records.append((move_bytes, score))

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, 'synthetic_selfplay.wtb')
    with open(path, 'wb') as f:
        # ヘッダ 16 バイト
        f.write(bytes([20, 26, 7, 11]))                # 作成日 (世紀,年,月,日)
        f.write(struct.pack('<I', len(records)))       # 対局数
        f.write(struct.pack('<H', 0))                  # レコード数 (.wtbでは未使用)
        f.write(struct.pack('<H', 2026))               # 対局年
        f.write(bytes([8, 0, 22, 0]))                  # 盤サイズ, 種別, 深さ, 予約
        for move_bytes, score in records:
            f.write(struct.pack('<HHH', 0, 0, 0))      # 大会/黒番/白番ID (ダミー)
            f.write(bytes([score, score]))             # 実スコア, 理論スコア
            padded = move_bytes + [0] * (60 - len(move_bytes))
            f.write(bytes(padded))
    print(f'wrote {len(records)} games to {path}')


# ---------------------------------------------------------------------------
# 学習と評価
# ---------------------------------------------------------------------------

def train(net, train_x, train_y, val_x, val_y, epochs, batch_size, lr):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(train_x))
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            x = train_x[idx].to(DEVICE)
            y = train_y[idx].to(DEVICE)
            loss = nn.functional.cross_entropy(net(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        net.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(val_x), 4096):
                x = val_x[i:i + 4096].to(DEVICE)
                y = val_y[i:i + 4096].to(DEVICE)
                correct += (net(x).argmax(dim=1) == y).sum().item()
        acc = correct / max(1, len(val_x))
        print(f'epoch {epoch}: train loss {total_loss / n_batches:.4f}  '
              f'val top-1 accuracy {acc:.1%}')


def policy_action(net, obs):
    """action_mask で無効手を弾いた上で方策ネットの最尤手を返す。"""
    mask = obs['action_mask'].flatten()
    with torch.no_grad():
        x = torch.as_tensor(obs['observation'], dtype=torch.float32,
                            device=DEVICE).unsqueeze(0)
        logits = net(x).squeeze(0).cpu().numpy()
    logits[mask == 0] = -np.inf
    return int(np.argmax(logits))


def evaluate(net, opponent_fn, games, label):
    """方策ネット vs opponent_fn で対戦評価 (手番は交互に持つ)。"""
    results = {'win': 0, 'lose': 0, 'draw': 0}
    env = OthelloEnv()
    for i in range(games):
        net_player = 'white' if i % 2 == 0 else 'black'
        obs, _ = env.reset()
        done = False
        while not done:
            player = env.current_player
            if player == net_player:
                action = policy_action(net, obs[player])
            else:
                action = opponent_fn(obs[player])
            obs, _, terminateds, _, _ = env.step({player: action})
            done = terminateds['__all__']
        diff = env.Board.count_white() - env.Board.count_black()
        if net_player == 'black':
            diff = -diff
        if diff > 0:
            results['win'] += 1
        elif diff < 0:
            results['lose'] += 1
        else:
            results['draw'] += 1
    total = sum(results.values())
    print(f"{label} ({total} games): win {results['win']} / lose {results['lose']} "
          f"/ draw {results['draw']}  (win rate {results['win'] / total:.1%})")


def run_evaluations(net, games):
    evaluate(net, OthelloEnv.random_agent, games, 'vs random')
    if os.path.exists(DQN_MODEL_PATH):
        from train_dqn import QNet, select_action
        qnet = QNet().to(DEVICE)
        qnet.load_state_dict(torch.load(DQN_MODEL_PATH, map_location=DEVICE))
        qnet.eval()
        evaluate(net, lambda obs: select_action(qnet, obs, 0.0), games, 'vs DQN')


def main():
    parser = argparse.ArgumentParser(
        description='Supervised policy training from WTHOR game records')
    parser.add_argument('--data-dir', default=DATA_DIR)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--winner-only', action='store_true',
                        help='勝者側の着手だけを学習に使う')
    parser.add_argument('--eval-games', type=int, default=200)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--make-synthetic', type=int, metavar='N',
                        help='WTHOR形式の合成自己対戦データを N 局生成して終了')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.make_synthetic:
        make_synthetic(args.data_dir, args.make_synthetic, seed=args.seed)
        return

    net = PolicyNet().to(DEVICE)

    if args.eval_only:
        net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        net.eval()
        run_evaluations(net, args.eval_games)
        return

    dataset = load_dataset(args.data_dir, winner_only=args.winner_only,
                           val_ratio=args.val_ratio, seed=args.seed)
    if dataset is None:
        print(f'no .wtb files found in {args.data_dir}\n'
              f'https://www.ffothello.org/informatique/la-base-wthor/ から\n'
              f'WTH_YYYY.ZIP をダウンロードして .wtb を置くか、\n'
              f'--make-synthetic N で動作確認用の合成データを生成してください。')
        sys.exit(1)

    train(net, *dataset, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr)
    torch.save(net.state_dict(), MODEL_PATH)
    print(f'saved model to {MODEL_PATH}')

    net.eval()
    run_evaluations(net, args.eval_games)


if __name__ == '__main__':
    main()
