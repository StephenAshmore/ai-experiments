import argparse

# Uses argparse to parse command line arguments:
def parse_arguments():
    parser = argparse.ArgumentParser(description='Run experiments on OpenAI\'s gym.')
    parser.add_argument('--model', dest='model', required=True,
        help='This determines which model to use, such as obs-policy',
        choices=('obs-policy', 'baseline'))
    parser.add_argument('--environment', dest='environment', required=True,
        help='This determines which environment or game to run the model against.',
        choices=('SpaceInvaders-v0', 'cartpole'))
    parser.add_argument('--seed', dest='seed',
        help='This determines what seed to initialize to.',
        type=int)
    parser.add_argument('--episodes', dest='episode_count', required=True,
        help='This determines how many episodes to train for.',
        type=int)
    parser.add_argument('--render', dest='render', type=int)
    parser.add_argument('--neurons', dest='neurons', type=int)
    parser.add_argument('--layers', dest='layers', type=int)
    parser.add_argument('--convs', dest='convs', type=int)
    return parser.parse_args()
