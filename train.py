# import os
# # Set the CUDA library path
# os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

# # Ensure the PATH includes the Ninja build system
# ninja_path = "/export/livia/home/vision/Abahri/anaconda3/envs/MATE/bin"
# os.environ['PATH'] = f"{ninja_path}:{os.environ.get('PATH', '')}"  


import os

print("PATH:", os.environ.get("PATH"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))

# Assuming the full path to the Ninja executable is as you specified:
ninja_path = "/export/livia/home/vision/Abahri/anaconda3/envs/MATE/bin/ninja" 
os.environ['PATH'] += os.pathsep + ninja_path


from tools import pretrain_run_net as pretrain
from tools import finetune_run_net as finetune   
from tools import ttt_rotnet as tttrotnet
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *  
import time
import os
import torch
from tensorboardX import SummaryWriter


def main(args):
    # only supporting these three datasts atm
    if args.dataset_name not in ['modelnet', 'scanobject', 'scanobject_nbg', 'partnet', 'shapenetcore', 'shapenet']:
        raise NotImplementedError

    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger=logger)
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size * 2
        config.dataset.val.others.bs = config.total_bs // world_size * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs
            # log
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank,
                             deterministic=args.deterministic)  # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()

    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold

    if args.dataset_name == 'scanobject' or args.dataset_name == 'scanobject_nbg':
        assert args.ckpts is not None  # because for scan object we only finetune from pretrained weights

    if args.train_tttrot:  # to train a model for rotation prediction
        tttrotnet(args, config, train_writer, val_writer)

    if args.jt or args.only_cls:
        pretrain(args, config, train_writer, val_writer)

    if args.finetune_model or args.scratch:
        finetune(args, config, train_writer, val_writer)


if __name__ == '__main__':
    args = parser.get_args()
    main(args)
