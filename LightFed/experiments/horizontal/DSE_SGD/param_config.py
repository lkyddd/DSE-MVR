import argparse
import logging
import os

import numpy as np
import torch
from experiments.datasets.data_distributer import DataDistributer
from lightfed.tools.funcs import consistent_hash, set_seed


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--comm_round', type=int, default=10000)

    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--eval_step_interval', type=int, default=1)

    parser.add_argument('--eval_batch_size', type=int, default=2048)

    parser.add_argument('--eval_on_full_test_data', type=lambda s: s == 'true', default=True)

    parser.add_argument('--gamma_list',
                        type=lambda s: [(int(item[0]), float(item[1])) for item in (t.split('=') for t in s.split(','))],
                        default='0=0.01,1000=0.1,4000=0.01,7000=0.001')

    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--tau', type=int, default=7, help='neighbor averaging interval')

    parser.add_argument('--model_type', type=str, default='Model_CIFAR10')

    parser.add_argument('--data_set', type=str, default='CIFAR-10')

    parser.add_argument('--data_partition_mode', type=str, default='iid', choices=['iid', 'non_iid_dirichlet'])

    parser.add_argument('--non_iid_alpha', type=float, default=10)

    parser.add_argument('--client_num', type=int, default=10)

    parser.add_argument('--device', type=torch.device, default='cuda')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--log_level', type=logging.getLevelName, default='INFO')

    parser.add_argument('--app_name', type=str, default='DLDSE_SGD')

    args = parser.parse_args()



    super_params = args.__dict__.copy()
    del super_params['log_level']
    super_params['device'] = super_params['device'].type
    ff = f"{args.app_name}-{consistent_hash(super_params, code_len=64)}.pkl"
    ff = f"{os.path.dirname(__file__)}/Result/{ff}"
    if os.path.exists(ff):
        print(f"output file existed, skip task")
        exit(0)

    args.data_distributer = _get_data_distributer(args)

    args.weight_matrix = _get_weight_matrix(args)

    return args


def _get_data_distributer(args):
    set_seed(args.seed + 5363)
    return DataDistributer(args)


def _get_weight_matrix(args):
    n = args.client_num
    wm = np.zeros(shape=(n, n))
    for i in range(n):
        wm[i][(i - 1 + n) % n] = 1 / 3
        wm[i][i] = 1 / 3
        wm[i][(i + 1 + n) % n] = 1 / 3

    assert np.allclose(wm, wm.T)
    return wm
