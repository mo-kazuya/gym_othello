"""
Snakeゲーム (pygame実装)

矢印キー（またはWASD）でヘビを操作し、エサ（赤）を food べてスコアを伸ばす
シンプルなSnakeゲーム。壁や自分自身にぶつかるとゲームオーバー。

操作方法:
    - 矢印キー / WASD : ヘビの移動方向を変更
    - スペース / Enter : ゲームオーバー後にリスタート
    - Esc / ウィンドウを閉じる : 終了

実行方法:
    python snake_game.py
"""

import random
import sys

import pygame

# --- 定数 ---
CELL_SIZE = 24            # 1マスのピクセルサイズ
GRID_WIDTH = 24          # 横方向のマス数
GRID_HEIGHT = 24         # 縦方向のマス数
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT
FPS = 10                 # 1秒あたりの移動回数（難易度）

# 色（RGB）
COLOR_BG = (18, 18, 18)
COLOR_GRID = (30, 30, 30)
COLOR_SNAKE_HEAD = (80, 220, 120)
COLOR_SNAKE_BODY = (60, 170, 90)
COLOR_FOOD = (230, 70, 70)
COLOR_TEXT = (240, 240, 240)
COLOR_OVERLAY = (0, 0, 0)

# 移動方向 (dx, dy)
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)


class SnakeGame:
    """Snakeゲームの状態と描画を管理するクラス。"""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 32)
        self.big_font = pygame.font.SysFont(None, 64)
        self.reset()

    def reset(self):
        """ゲーム状態を初期化する。"""
        center = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
        # ヘビは長さ3で中央から左向きに配置（先頭が先頭要素）
        self.snake = [center, (center[0] - 1, center[1]), (center[0] - 2, center[1])]
        self.direction = RIGHT
        self.next_direction = RIGHT
        self.food = self._spawn_food()
        self.score = 0
        self.game_over = False

    def _spawn_food(self):
        """ヘビと重ならない位置にエサをランダム配置する。"""
        empty_cells = [
            (x, y)
            for x in range(GRID_WIDTH)
            for y in range(GRID_HEIGHT)
            if (x, y) not in self.snake
        ]
        if not empty_cells:
            return None  # 盤面が埋まった（クリア）
        return random.choice(empty_cells)

    def _handle_input(self):
        """キー入力を処理する。"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._quit()
                if self.game_over:
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        self.reset()
                    continue
                # 進行方向の変更（逆走は禁止）
                if event.key in (pygame.K_UP, pygame.K_w) and self.direction != DOWN:
                    self.next_direction = UP
                elif event.key in (pygame.K_DOWN, pygame.K_s) and self.direction != UP:
                    self.next_direction = DOWN
                elif event.key in (pygame.K_LEFT, pygame.K_a) and self.direction != RIGHT:
                    self.next_direction = LEFT
                elif event.key in (pygame.K_RIGHT, pygame.K_d) and self.direction != LEFT:
                    self.next_direction = RIGHT

    def _update(self):
        """ヘビを1マス進め、衝突・エサ取得を判定する。"""
        if self.game_over:
            return

        self.direction = self.next_direction
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # 壁との衝突判定
        if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
            self.game_over = True
            return

        # 自分自身との衝突判定（末尾は今回移動で空くので除外）
        if new_head in self.snake[:-1]:
            self.game_over = True
            return

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.food = self._spawn_food()
            if self.food is None:
                # 盤面をすべて埋めた＝クリア扱いでゲーム終了
                self.game_over = True
        else:
            self.snake.pop()  # エサを取っていなければ末尾を削除して長さ維持

    def _draw_grid(self):
        """背景グリッドを描画する。"""
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, COLOR_GRID, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, COLOR_GRID, (0, y), (SCREEN_WIDTH, y))

    def _draw_cell(self, pos, color):
        """1マスを塗りつぶす。"""
        rect = pygame.Rect(
            pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE
        )
        pygame.draw.rect(self.screen, color, rect.inflate(-2, -2), border_radius=4)

    def _draw(self):
        """全体を描画する。"""
        self.screen.fill(COLOR_BG)
        self._draw_grid()

        # エサ
        if self.food is not None:
            self._draw_cell(self.food, COLOR_FOOD)

        # ヘビ（先頭とそれ以外で色を変える）
        for i, segment in enumerate(self.snake):
            color = COLOR_SNAKE_HEAD if i == 0 else COLOR_SNAKE_BODY
            self._draw_cell(segment, color)

        # スコア表示
        score_text = self.font.render(f"Score: {self.score}", True, COLOR_TEXT)
        self.screen.blit(score_text, (8, 6))

        # ゲームオーバー表示
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(160)
            overlay.fill(COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))

            cleared = self.food is None
            title = "CLEAR!" if cleared else "GAME OVER"
            title_surf = self.big_font.render(title, True, COLOR_TEXT)
            title_rect = title_surf.get_rect(
                center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30)
            )
            self.screen.blit(title_surf, title_rect)

            info = self.font.render(
                f"Score: {self.score}   Press SPACE to restart", True, COLOR_TEXT
            )
            info_rect = info.get_rect(
                center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20)
            )
            self.screen.blit(info, info_rect)

        pygame.display.flip()

    def _quit(self):
        pygame.quit()
        sys.exit()

    def run(self):
        """メインループ。"""
        while True:
            self._handle_input()
            self._update()
            self._draw()
            self.clock.tick(FPS)


def main():
    SnakeGame().run()


if __name__ == "__main__":
    main()
