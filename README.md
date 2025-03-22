# gym_othello
petting zooインタフェースに対応したオセロゲーム

## インストール

```
pip install pygame
pip install ray[rllib]
pip install git+https://github.com/mo-kazuya/gym_othello/tree/main/othello/envs
```


## 使い方

```
from gym_othello.othello.envs import OthelloEnv
env = OthelloEnv()
env.reset()
```

```
from gym_othello.othello.envs import OthelloEnv
env = OthelloEnv(render_mode='human')
env.reset()
```