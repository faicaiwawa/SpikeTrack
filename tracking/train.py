import argparse
import os


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple"], default="multiple",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, help="number of GPUs per node")
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)

    args = parser.parse_args()

    return args


def main():
    # modified in nov.2025, from torch.distributed.launch to torchrun

    args = parse_args()
    if args.mode == "single":
        train_cmd = "torchrun --nproc_per_node 1 lib/train/run_training.py " \
                    "--script %s --config %s --save_dir %s --use_lmdb %d " \
                    % (args.script, args.config, args.save_dir, args.use_lmdb)
    elif args.mode == "multiple":
        train_cmd = "torchrun --nproc_per_node %d lib/train/run_training.py " \
                    "--script %s --config %s --save_dir %s --use_lmdb %d " \
                    % (args.nproc_per_node, args.script, args.config, args.save_dir, args.use_lmdb)
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")
    print(train_cmd)
    os.system(train_cmd)


if __name__ == "__main__":
    main()
