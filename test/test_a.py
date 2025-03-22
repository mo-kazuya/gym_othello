import unittest
import othello.envs.othello_env as othello_env

class TestBoard( unittest.TestCase ):
    def test_Board(self):
        board = othello_env.Board()
        board.initialize_board()

        self.assertEqual( board.count_white(), 2 )
        self.assertEqual( board.count_black(), 2 )

        valid_moves = board.get_valid_moves( board.current_player )

class TestEnv( unittest.TestCase ):
    def test_Env(self):
        env = othello_env.OthelloEnv()
        obs,info = env.reset()

        self.assertEqual( env.current_player , 'white' )

        self.assertEqual( env.num_agents, 2 )
        self.assertEqual( env.max_num_agents, 2 )

if __name__ == '__main__':
    unittest.main()


