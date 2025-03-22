import gymnasium as gym
import numpy as np
from typing import Optional
import pygame
import copy
import random
import os

from logging import getLogger, StreamHandler, DEBUG, INFO

BLOCK_SIZE=20
WIDTH = 640
HEIGHT = 480

COLOR_WHITE = (255,255,255)
COLOR_BLACK = (0,0,0)

PLAYER_WHITE = 1
PLAYER_BLACK = 2

# オセロゲームの盤面を表すクラス
class Board:
    def __init__(self, render_mode=None):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.current_player = PLAYER_WHITE

        self.initialize_board()

        self.render_mode = render_mode
        if self.render_mode == "human" :
            if pygame.font.get_init() == False:
                pygame.font.init()

            font_path = os.path.join( os.path.dirname(__file__) , 'atari.ttf' )
            # self.font = pygame.font.Font('/work/misc/othello2/gym_othello2/atari.ttf',25)
            self.font = pygame.font.Font(font_path,25)

    def initialize_board(self):
        self.current_player = PLAYER_WHITE
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.board[3][3] = self.board[4][4] = PLAYER_BLACK 
        self.board[3][4] = self.board[4][3] = PLAYER_WHITE
    
    # 白の数を数える
    def count_white(self):
        return sum([row.count(PLAYER_WHITE) for row in self.board])
    
    # 黒の数を数える
    def count_black(self):
        return sum([row.count(PLAYER_BLACK) for row in self.board])
    
    # 駒を置ける場所のリストを返す
    def get_valid_moves(self, player):
        moves = []
        for row in range(8):
            for col in range(8):
                if self.is_valid_move(col, row, player):
                    moves.append((col, row))
        return moves
    
    def get_board(self):
        return self.board

    def draw_board(self, screen):

        cell_size = 60

        pygame.draw.rect(
            screen, 
            (0,255,0), 
            pygame.Rect( 0, 0, 8*cell_size, 8*cell_size) 
        )

        for row in range(8):
            for col in range(8):
                # マス目を描画
                pygame.draw.rect(
                    screen, 
                    (0,0,0), 
                    pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size), 
                    2
                )
                # 駒を描画
                if self.board[row][col] != 0:
                    pygame.draw.circle(
                        screen, 
                        COLOR_WHITE if self.board[row][col] == PLAYER_WHITE else COLOR_BLACK , 
                        (col * cell_size + cell_size//2 , row * cell_size + cell_size//2), 
                        25
                    )

        # 駒を置ける場所を表示
        for col,row in self.get_valid_moves(self.current_player):
            pygame.draw.circle(
                screen, 
                (128,128,128), 
                (col * cell_size + cell_size//2 , row * cell_size + cell_size//2), 
                25, 2
            )

        # スコア表示
        text1 = self.font.render( f'WHITE: {self.count_white()}',True,COLOR_WHITE)
        text2 = self.font.render( f"BLACK: {self.count_black()}",True,COLOR_WHITE)
        screen.fill((0,0,0),rect=pygame.Rect(8*61,0,8*61,27))
        screen.blit(text1,[8*61, 0])
        screen.fill((0,0,0),rect=pygame.Rect(8*61,27,8*61,27))
        screen.blit(text2,[8*61,27])

        screen.fill((0,0,0),rect=pygame.Rect(8*61,54,8*61,27))
        if self.is_game_over():
            if self.count_white() > self.count_black():
                text = self.font.render( f"WHITE WINS",True,COLOR_WHITE)
            elif self.count_white() < self.count_black():
                text = self.font.render( f"BLACK WINS",True,COLOR_WHITE)
            else:
                text = self.font.render( f"DRAW",True,COLOR_WHITE)
            screen.blit(text,[8*61,54])

        pygame.display.flip()

    # 駒を置けるかどうかを判定
    def is_valid_move(self, col, row, player):
        if self.board[row][col] != 0:
            return False
        opponent = PLAYER_WHITE if player == PLAYER_BLACK else PLAYER_BLACK
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for d in directions:
            x, y = col, row
            x += d[0]
            y += d[1]
            if x < 0 or x >= 8 or y < 0 or y >= 8 or self.board[y][x] != opponent:
                continue
            x += d[0]
            y += d[1]
            while x >= 0 and x < 8 and y >= 0 and y < 8:
                if self.board[y][x] == player:
                    return True
                if self.board[y][x] == 0:
                    break
                x += d[0]
                y += d[1]
        return False

    # 駒を置く
    # True: 駒を置けた, False: 駒を置けなかった
    def place_piece(self, col, row, player):
        res = False
        if not self.is_valid_move(col, row, player):
            res = False
        else:
            self.board[row][col] = player
            self.flip_pieces(col, row, player)
            res = True
        
        # 駒を置けた場合はターンを交代
        if res == True:
            self.current_player = PLAYER_WHITE if self.current_player == PLAYER_BLACK else PLAYER_BLACK
            # パスの処理
            if len(self.get_valid_moves(self.current_player)) == 0:
                # self.logger.debug("Pass")
                self.current_player = PLAYER_WHITE if self.current_player == PLAYER_BLACK else PLAYER_BLACK
        else:
            self.current_player = self.current_player
        
        return res
        
    # 駒を裏返す
    def flip_pieces(self, col, row, player):
        opponent = PLAYER_WHITE if player == PLAYER_BLACK else PLAYER_BLACK
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for d in directions:
            x, y = col, row
            x += d[0]
            y += d[1]
            if x < 0 or x >= 8 or y < 0 or y >= 8 or self.board[y][x] != opponent:
                continue
            positions = []
            positions.append((x, y))
            x += d[0]
            y += d[1]
            while x >= 0 and x < 8 and y >= 0 and y < 8:
                if self.board[y][x] == player:
                    for pos in positions:
                        # self.board[pos[0]][pos[1]] = player
                        self.board[pos[1]][pos[0]] = player
                    break
                if self.board[y][x] == 0:
                    break
                positions.append((x, y))
                x += d[0]
                y += d[1]
        return
    
    # ゲーム終了判定
    def is_game_over(self):
        return len(self.get_valid_moves(PLAYER_WHITE)) == 0 and len(self.get_valid_moves(PLAYER_BLACK)) == 0

# オセロゲームのplayerを表すクラス
class Player:
    def __init__(self, name, color):
        self.color = color
        self.name = name

    def get_move(self, board):
        # Implement logic to get the move
        return 0, 0  # Placeholder for actual
    
# from ray.rllib.env.multi_agent_env import MultiAgentEnv
class OthelloEnv(gym.Env):

    def __init__(self, render_mode=None, random_offset=None):
        super().__init__()


        self.logger = getLogger(__name__)

        self.Board = Board(render_mode=render_mode)

        self.before_bord = self.Board.get_board()

        self.agents = self.possible_agents = ["white", "black"]

        self.observation_spaces = {
            "white": gym.spaces.Dict(
                {
                    "observation": gym.spaces.Box(0,1,(2,8,8),dtype=np.int8),
                    "action_mask": gym.spaces.Box( 0, 1, (8,8), dtype=np.int8) ,
                }
            ),
            "black": gym.spaces.Dict(
                {
                    "observation": gym.spaces.Box(0,1,(2,8,8),dtype=np.int8),
                    "action_mask": gym.spaces.Box( 0, 1, (8,8), dtype=np.int8) ,
                }
            ),
        }

        self.action_spaces = {
            "white": gym.spaces.Discrete(64),
            "black": gym.spaces.Discrete(64),
        }

        self.w = int(WIDTH/BLOCK_SIZE)
        self.h = int(HEIGHT/BLOCK_SIZE)

        self.display = None
        self.render_mode = render_mode
        self.random_offset = random_offset
        self.current_player = None

    def get_observation_space(self, agent_id) -> gym.Space:
        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id ) -> gym.Space:
        return self.action_spaces[agent_id]

    @property
    def num_agents(self) -> int:
        return len(self.agents)
    
    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)

    def step(self, action):
        self.logger.debug(f"step action:{action}")
        self.logger.debug(f"valid mode:{self.Board.get_valid_moves(self.Board.current_player)}")
        
        rewards = {
            self._get_player(): 0.0,
            self._get_opponent(): 0.0,
        }
        terminateds = {"__all__": False}

        self.before_bord = copy.deepcopy( self.Board.get_board() )

        # プレイヤーのチェック
        if self._get_player() not in action:
            self.logger.info("Invalid player")
            rewards[self._get_player()] = -10.0
            terminateds["__all__"] = True
            return self._get_obs(), rewards, terminateds, {}, self._get_info()

        pos = action[self._get_player()]
        pos_col = pos%8
        pos_row = pos//8

        # 駒を置けるかどうかのチェック
        if not self.Board.is_valid_move(pos_col, pos_row, self.Board.current_player):
            self.logger.info("Invalid move")
            rewards[self._get_player()] = -10.0
            terminateds["__all__"] = True
            return self._get_obs(), rewards, terminateds, {}, self._get_info()

        # ボードに駒を置く
        res = Board.place_piece(
            self.Board, 
            pos_col,  # col
            pos_row,  # row
            self.Board.current_player    
        )
        self.current_player = self._get_player()

        # 4隅に駒を置いた場合
        if (
            ( pos_col == 0 and pos_row == 0 ) 
            or ( pos_col == 7 and pos_row == 0 )
            or ( pos_col == 0 and pos_row == 7 )
            or ( pos_col == 7 and pos_row == 7 )
        ) :
            rewards[self._get_player] = 0.25
            rewards[self._get_opponent] = -0.25

        # ゲーム終了判定
        if self.Board.is_game_over():
            rewards['white'] = (self.Board.count_white() - self.Board.count_black() + 4 ) // 5.0
            rewards['black'] = (self.Board.count_black() - self.Board.count_white() + 4 ) // 5.0
            # if self.Board.count_white() > self.Board.count_black():
            #     self.logger.info("White wins")
            #     rewards["white"] = 5.0
            #     rewards["black"] = -5.0
            # elif self.Board.count_white() < self.Board.count_black():
            #     self.logger.info("Black wins")
            #     rewards["black"] = 5.0
            #     rewards["white"] = -5.0
            # else:
            #     self.logger.info("Draw")
            #     rewards["black"] = 1.0
            #     rewards["white"] = 1.0
            terminateds["__all__"] = True

        # obs, rewards, terminateds, truncateds, infos
        return self._get_obs(), rewards, terminateds, {}, self._get_info()

    def _get_obs(self):
        player = "white" if self.Board.current_player == PLAYER_WHITE else "black"

        white_board = np.int8(np.array(self.Board.get_board()) == PLAYER_WHITE)
        black_board = np.int8(np.array(self.Board.get_board()) == PLAYER_BLACK)

        my_board = white_board if player == "white" else black_board
        opponent_board = black_board if player == "white" else white_board

        action_mask = np.zeros((8,8),dtype=np.int8)
        # 駒を置ける場所に1をセット
        valid_moves = np.array(self.Board.get_valid_moves( self.Board.current_player))

        if len(valid_moves) > 0:
            cols,rows = valid_moves.T
            action_mask[rows,cols] = 1
        # action_mask[ 
        #     tuple(np.array(self.Board.get_valid_moves(self.Board.current_player)).T) 
        # ] = 1

        return {
            player : {
                "observation": np.stack((my_board, opponent_board)),
                "action_mask": action_mask,
            }
        }

    def _get_info(self):
        player = "white" if self.Board.current_player == PLAYER_WHITE else "black"

        return {
            player : {
                "white_count": self.Board.count_white(),
                "black_count": self.Board.count_black(),
                "board": self.Board.get_board(),
                "before_board": self.before_bord,
            }
        }
    
    def _get_player(self):
        return "white" if self.Board.current_player == PLAYER_WHITE else "black"
    
    def _get_opponent(self):
        return "black" if self.Board.current_player == PLAYER_WHITE else "white"
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.logger.debug("reset")
        super().reset(seed=seed)

        self.Board.initialize_board()
        self.Board.current_player = PLAYER_WHITE
        self.current_player = self._get_player()

        observation = self._get_obs()
        info = self._get_info()


        offset = 0
        if options is not None and 'offset' in options:
            offset = options['offset']
        elif self.random_offset is not None and self.random_offset > 0:
            offset = int(random.random()*self.random_offset)

        if offset > 0:
            for i in range( offset ):
                action = self.random_agent( observation[self.current_player] )
                observation,rew,term,trunc,info =  self.step( {self.current_player:action} )


        if self.render_mode == "human" :
            if pygame.display.get_init() == False:
                pygame.display.init()
            self.display = pygame.display.set_mode((self.w*BLOCK_SIZE,self.h*BLOCK_SIZE))
            pygame.display.set_caption('Othello')
            self.clock = pygame.time.Clock()

            if pygame.font.get_init() == False:
                pygame.font.init()
            self.font = pygame.font.Font('/work/misc/othello2/gym_othello2/atari.ttf',25)
        
            self.Board.draw_board(self.display)
        
        else:
            self.display = None

        # if pygame.font.get_init() == False:
        #     pygame.font.init()


        return observation, info

    def render(self):
        if self.render_mode == "human" and self.display is not None:
            self.Board.draw_board(self.display)
            # self.clock.tick(60)
            pygame.display.update()
        else:
            return self.Board.get_board() 

    def close(self):
        pygame.quit()

    @classmethod
    def random_agent(cls,obs):
        act_mask = obs['action_mask']
        act = np.random.choice( np.flatnonzero(act_mask) ) 
        return act    
    
    @classmethod
    def human_agent(cls,obs):
        clock = pygame.time.Clock()

        while True:
            flg = False
            for evt in pygame.event.get():
                if evt.type == pygame.MOUSEBUTTONDOWN:
                    col = evt.pos[0] // 60
                    row = evt.pos[1] // 60
                    flg = True

            if flg:
                act = col+row*8
                act_mask = obs['action_mask']
                valid_acts = np.flatnonzero(act_mask)
                if act in  valid_acts:
                    break
            clock.tick(10)

        return act    




