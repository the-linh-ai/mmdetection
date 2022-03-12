#!/usr/bin/python3

import os
import sys
import argparse
import subprocess
import random
import socket


def scan_port(ip, port_list):
    def check_port(ip, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            result = s.connect_ex((ip, int(port)))
            if result == 0:
                return False
            else:
                return True
        except:
            return False
        finally:
            s.close()
    random.shuffle(port_list)
    for port in port_list:
        if check_port(ip, port):
            return port
    return None


def init_workdir():
    HAGGS_ROOT = os.path.dirname(os.path.abspath(__file__))
    os.chdir(HAGGS_ROOT)
    sys.path.insert(0, HAGGS_ROOT)


def set_environ():
    # OS env variables
    if 'HADOOP_ROOT_LOGGER' not in os.environ:
        # disable hdfs verbose logging
        os.environ['HADOOP_ROOT_LOGGER'] = 'ERROR,console'

    # disable hdfs verbose logging
    os.environ['LIBHDFS_OPTS'] = '-Dhadoop.root.logger={}'.format(
        os.environ['HADOOP_ROOT_LOGGER'])
    # set JVM heap memory
    os.environ['LIBHDFS_OPTS'] += '-Xms512m -Xmx10g ' + \
        os.environ['LIBHDFS_OPTS']
    # set KRB5CCNAME for hdfs
    os.environ['KRB5CCNAME'] = '/tmp/krb5cc'

    # disable TF verbose logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # fix known issues for pytorch-1.5.1 accroding to
    # https://blog.exxactcorp.com/pytorch-1-5-1-bug-fix-release/
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    # set NCCL envs for disributed communication
    os.environ['NCCL_IB_GID_INDEX'] = '3'
    os.environ['NCCL_IB_DISABLE'] = '0'
    os.environ['NCCL_IB_HCA'] = 'mlx5_2:1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'

    os.environ['ARNOLD_FRAMEWORK'] = 'pytorch'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Haggs Launcher')
    parser.add_argument('--launch', type=str, default='train.py',
                        help='Specify launcher script.')
    parser.add_argument('--dist', type=int, default=1,
                        help='Whether start by torch.distributed.launch.')
    parser.add_argument('--np', type=int, default=-1,
                        help='number of processes per node')
    parser.add_argument('--nn', type=int, default=-1,
                        help='number of workers')
    parser.add_argument('--port', type=int, default=-1,
                        help='master port for communication')
    args, other_args = parser.parse_known_args()

    init_workdir()
    set_environ()

    # get arnold env
    master_address = os.getenv('METIS_WORKER_0_HOST')
    if master_address is None:
        # Not in an Arnold machine
        print("Not in an Arnold machine")
        if args.np <= 0:
            raise RuntimeError(
                "If not in an Arnold machine, --np (number of processes) must "
                f"be specified explicitly"
            )
        master_address = socket.gethostbyname(socket.gethostname())
        num_processes_per_worker = args.np
        num_workers = 1
        worker_rank = 0

    else:
        num_processes_per_worker = int(os.getenv('ARNOLD_WORKER_GPU'))
        num_workers = int(os.getenv('ARNOLD_WORKER_NUM'))
        worker_rank = int(os.getenv('ARNOLD_ID'))

    assert num_workers == 1, f"`al_seg` only supports single-node training"

    if args.np > 0:
        assert args.np <= num_processes_per_worker
        num_processes_per_worker = args.np
    if args.nn > 0:
        assert args.nn <= num_workers
        num_workers = args.nn
    # get port
    if args.port > 0:
        master_port = args.port
    else:
        if num_workers == 1:
            master_port = scan_port(
                master_address, port_list=list(range(40001, 45000)))
            if master_port is None:
                print('Error: Can not find a valid port!')
                sys.exit(-1)
        else:
            master_port = int(os.getenv('METIS_WORKER_0_PORT'))

    if args.dist >= 1:
        print(f'Start {args.launch} by torch.distributed.launch with port {master_port}!', flush=True)
        cmd = f'python3 -m torch.distributed.launch \
                --nproc_per_node={num_processes_per_worker} \
                --master_port={master_port} \
                {args.launch}'
    else:
        print(f'Start {args.launch}!', flush=True)
        cmd = f'python3 {args.launch}'

    for argv in other_args:
        cmd += f' {argv}'

    exit_code = subprocess.call(cmd, shell=True)
    sys.exit(exit_code)
