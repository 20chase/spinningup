from spinup.utils.run_utils import ExperimentGrid
from spinup import sac_tf1
import tensorflow as tf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    eg = ExperimentGrid(name='sac_pos_skipframe_10')
    eg.add('env_name', ['AntBulletPositionControlEnv-v0', 
                        'Walker2DBulletPositionControlEnv-v0', 
                        'HalfCheetahBulletPositionControlEnv-v0', 
                        'HopperBulletPositionControlEnv-v0'
                        ])
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 200)
    eg.add('skip_frames', 10)
    eg.run(sac_tf1)