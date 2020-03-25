from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_tf1
import tensorflow as tf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    eg = ExperimentGrid(name='ppo_pos')
    eg.add('env_name', ['AntBulletPositionControlEnv-v0', 
                        'Walker2DBulletPositionControlEnv-v0', 
                        'HalfCheetahBulletPositionControlEnv-v0', 
                        'HopperBulletPositionControlEnv-v0',
                        'HumanoidBulletPositionControlEnv-v0'
                        ])
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 300)
    eg.add('steps_per_epoch', 8000)
    eg.run(ppo_tf1, num_cpu=args.cpu, datestamp=True)