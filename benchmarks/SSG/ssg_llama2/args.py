import argparse
import sys

def get_args(mode):
    common_parser = argparse.ArgumentParser()
    # ======================= data ======================= #
    common_parser.add_argument("--task", type=str, choices=['l2p', 'p2l'])
    
    common_parser.add_argument("--paper", action='store_true')
    common_parser.add_argument("--subset", action='store_true')

    common_parser.add_argument("--seed", type=int, default=42)
    common_parser.add_argument("--exp", type=str, default="try")

    common_args = common_parser.parse_known_args()[0]

    return_obj = (common_args, ) # there use parse_known_args
    if mode == 'train':
        return_obj += (_train_args(common_parser), )
    elif mode == 'generate':
        return_obj += (_generate_args(common_parser), )
    else:
        raise NotImplementedError

    return return_obj # the second one contains the first one


def _train_args(parser):
    # ======================= train ======================= #
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--train_per_gpu", type=int, default=2)
    parser.add_argument("--val_per_gpu", type=int, default=2)
    # ---------- lr ---------- #
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=0)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    # ------------------------ #
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--n_epochs", type=int, default=3)
    # ======================= others ======================= #

    args = parser.parse_args()

    assert args.train_batch_size % args.train_per_gpu == 0
    args.gradient_accumulation_steps = args.train_batch_size // args.train_per_gpu
    
    return args

def _generate_args(parser):
    # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L129-L137
    parser.add_argument("--max_gen_len", type=int, default=300)

    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--method",  default='sample', choices=['greedy_search', 'sample'])

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    print(get_args('train'))