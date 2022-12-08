import argparse
import os
import shutil
import time
import math
import random

import torch
import torch.distributed as dist
import torchvision.models as models
import torch.nn as nn
import numpy as np
from dist_utils import get_logger, DistSummaryWriter
from datetime import datetime
from models import get_vit
from data import make_data_loader
import ml_collections
# from test_tube import Experiment
from util import AverageMeter, AveragePrecisionMeter, WarmupCosineSchedule
# from torch.cuda.amp import GradScaler, autocast

def parse():
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name])) + ['swin_large_patch4_window12_384', 'fcanet50', 'fcanet101', 'fcanet152', 'ViT-B_16', 'resnet101']

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('--image-size', '-i', default=448, type=int)
    parser.add_argument('--gpus', default='0,1,2,3', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
    parser.add_argument('-w', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--epoch_step', default=[30, 40], type=int, nargs='+', 
                        help='number of epochs to change learning rate')
    parser.add_argument('--train_steps', default=10000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.5, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--lrp', '--learning-rate_p', default=0.1, type=float,
                        metavar='LR', help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--max_clip_grad_norm', default=10.0, type=float, metavar='M', 
                        help='max_clip_grad_norm')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--evaluate_model', type=str, default=None, help='the model for evaluation')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    parser.add_argument('--dali_cpu', action='store_true',
                        help='Runs CPU based version of DALI pipeline.')
    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    parser.add_argument('--opt-level', type=str, default=None)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    parser.add_argument('-t', '--test', action='store_true',
                        help='Launch test mode with preset arguments')
    parser.add_argument('--work_dir', type=str, default = './augments_DP')
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--data_root_dir', type=str, default='/data/datasets')
    parser.add_argument('--data', metavar='NAME', help='dataset name (e.g. COCO2014')
    parser.add_argument('--model_name', type=str, default='ADD_GCN')
    parser.add_argument('--crop_size', default=448, type=int, metavar='N',
                        help='crop_size')
    parser.add_argument('--scale_size', default=600, type=int, metavar='N',
                        help='scale_size')
    args = parser.parse_args()
    return args

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def print_params(args, prtf=print):
        opt = vars(args)
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(opt.items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


def main():
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    global best_prec1, args, logger
    best_prec1 = 0
    args = parse()
    args.gpus = parse_gpus(args.gpus)

    # if not len(args.data):
    #     raise Exception("error: No data set provided")

    best_prec1 = 0
    torch.set_printoptions(precision=10)
    setup_seed(0)

    args.gpu = args.gpus[0]

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda")

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    args.work_dir = os.path.join(args.work_dir, time_stamp + args.arch + '&'+ args.note)
    if not args.evaluate:
        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir)
        # logger = DistSummaryWriter(args.work_dir)
        # exp = Experiment(name=args.note,
        #          debug=False,
        #          save_dir=args.work_dir)
    # print(dist.get_rank())
    logger = get_logger(os.path.join(args.work_dir, "{}.log".format(args.note)))
    # print('hhhhhhhhhhhh')
    print_params(args, logger.info)

    if args.opt_level is not None and dist.get_rank() == 0:
        logger.info("opt_level = {}".format(args.opt_level))
    if args.keep_batchnorm_fp32 is not None and dist.get_rank() == 0:
        logger.info("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    if args.loss_scale is not None and dist.get_rank() == 0:
        logger.info("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))
    logger.info("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))
    # exp.log({'\nCUDNN VERSION': torch.backends.cudnn.version()})

    # Data loading code
    is_train = True
    train_loader, val_loader, num_classes = make_data_loader(args, is_train=is_train)

    # create model
    if args.pretrained and dist.get_rank() == 0:
        logger.info("=> using pre-trained model '{}'".format(args.arch))
        if args.arch == 'fcanet34':
            model = fcanet34(pretrained=True)
        elif args.arch == 'fcanet50':
            model = fcanet50(pretrained=True)
        elif args.arch == 'fcanet101':
            model = fcanet101(pretrained=True)
        elif args.arch == 'fcanet152':
            model = fcanet152(pretrained=True)
        else:
            model = models.__dict__[args.arch](pretrained=True)
    else:
        logger.info("=> creating model '{}'".format(args.arch))
        if args.arch == 'fcanet34':
            model = fcanet34()
        elif args.arch == 'fcanet50':
            model = fcanet50()
        elif args.arch == 'fcanet101':
            model = fcanet101()
        elif args.arch == 'fcanet152':
            model = fcanet152()
        elif args.arch == 'resnet101':
            model = get_model(num_classes, args)
        elif args.arch == 'ViT-B_16':
            model = get_vit(num_classes)
        else:
            model = models.__dict__[args.arch]()

    
    model = model.to(device)

    # Scale learning rate based on global batch size
    args.lr = args.lr#*float(args.batch_size*args.world_size)/16.
    
    # optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp), 
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), 
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.get_config_optim(args.lr, 10), 
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), 
    #                              eps=_C.TRAIN.OPTIMIZER.EPS, 
    #                              betas=_C.TRAIN.OPTIMIZER.BETAS,
    #                              lr=args.lr,
    #                              weight_decay=args.weight_decay)
    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    model = torch.nn.DataParallel(model, device_ids=args.gpus)
    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))

                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))
        resume()
    if args.evaluate:
        assert args.evaluate_model is not None
        logger.info("=> loading checkpoint '{}' for eval".format(args.evaluate_model))
        checkpoint = torch.load(args.evaluate_model, map_location = lambda storage, loc: storage.cuda(args.gpu))
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
        else:
            state_dict_with_module = {}
            for k,v in checkpoint.items():
                state_dict_with_module['module.'+k] = v
            model.load_state_dict(state_dict_with_module)



    criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    # criterion = CrossEntropyLabelSmooth().cuda()

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    scheduler = WarmupCosineSchedule(optimizer,
                                    warmup_steps=int(args.train_steps*0.05), 
                                    t_total=args.train_steps)

    args.epochs = args.train_steps//len(train_loader)
    meters_train1 = initialize_meters()
    meters_train2 = initialize_meters()
    meters_val1 = initialize_meters()
    meters_val2 = initialize_meters()
    for epoch in range(args.start_epoch, args.epochs):
        on_start_epoch(meters_train1)
        on_start_epoch(meters_val1)
        on_start_epoch(meters_train2)
        on_start_epoch(meters_val2)
        # train for one epoch
        # lr_now = adjust_learning_rate(optimizer, epoch, args.epoch_step)
        # print('lr_now', lr_now)
        _,_= train(train_loader, model, criterion, optimizer, epoch, scheduler, meters_train1, meters_train2, device)
        torch.cuda.empty_cache()
        # evaluate on validation set
        map_1, map_2 = validate(val_loader, model, criterion, epoch, meters_val1, meters_val2, device)

        map_1 = torch.tensor(map_1)
        map_2 = torch.tensor(map_2)
        # remember best prec@1 and save checkpoint
        map_ = torch.max(map_1, map_2)
        is_best = map_ > best_prec1
        best_prec1 = max(map_, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_score': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, work_dir = args.work_dir)
        if epoch == args.epochs - 1 and dist.get_rank() == 0:
            logger.info('##Best Top-1 {0}\n'
                  '##Perf  {2}'.format(
                  best_prec1,
                  args.batch_size / total_time.avg))
            with open(os.path.join(args.work_dir, 'res.txt'), 'w') as f:
                f.write('arhc: {0} \n best_prec1 {1}'.format(args.arch+args.note, best_prec1))

def initialize_meters():
    meters = {}
    # meters
    meters['loss'] = AverageMeter('loss')
    meters['ap_meter'] = AveragePrecisionMeter()
    # time measure
    meters['batch_time'] = AverageMeter('batch_time')
    meters['data_time'] = AverageMeter('data_time')
    return meters

def reset_meters(meters):
    for k, v in meters.items():
        meters[k].reset()

def on_start_epoch(meters):
    reset_meters(meters)


def train(train_loader, model, criterion, optimizer, epoch, scheduler, meters1, meters2, device):
    st_time = time.time()
    model.train()
    Sig = torch.nn.Sigmoid()
    preds_regular1 = []
    targets_1 = []
    preds_regular2 = []
    targets_2 = []
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time = time.time() - st_time
        meters1['data_time'].update(data_time)

        # inputs, targets, targets_gt, filenames = self.on_start_batch(data)
        inputs = data['image']
        targets = data['target']

        # for voc
        labels = targets.clone()
        targets[targets==0] = 1
        targets[targets==-1] = 0
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs1, outputs2 = model(inputs, device)
        loss1 = criterion(outputs1, targets)
        loss2 = criterion(outputs2, targets)
        loss = loss1+loss2
        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=args.max_clip_grad_norm)
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - st_time
        meters1['batch_time'].update(batch_time)
        meters1['ap_meter'].add(outputs1.data, labels.data, data['name'])
        meters2['ap_meter'].add(outputs2.data, labels.data, data['name'])
        st_time = time.time()
        # torch.cuda.synchronize()
        reduced_loss1 = loss.data
        meters1['loss'].update(to_python_float(reduced_loss1), inputs.size(0))
        reduced_loss2 = loss1.data
        meters2['loss'].update(to_python_float(reduced_loss2), inputs.size(0))

        output_regular1 = Sig(outputs1).cpu()
        preds_regular1.append(output_regular1.cpu().detach())
        targets_1.append(targets.cpu().detach())

        output_regular2 = Sig(outputs2).cpu()
        preds_regular2.append(output_regular2.cpu().detach())
        targets_2.append(targets.cpu().detach())

        if i%args.print_freq == 0:
            logger.info('{}, {} Epoch, {} Iter, Loss_all: {:.4f}, Loss1: {:.4f}, Data time: {:.4f}, Batch time: {:.4f}'.format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  epoch+1, i, 
                    meters1['loss'].value(), meters2['loss'].value(), meters1['data_time'].value(), 
                    meters1['batch_time'].value()))

    # ap = meters['ap_meter'].value()
    # if args.local_rank == 0 and dist.get_rank() == 0:
    #     logger.info('ap', ap)
    map1 = mAP(torch.cat(targets_1).numpy(), torch.cat(preds_regular1).numpy())
    map2 = mAP(torch.cat(targets_2).numpy(), torch.cat(preds_regular2).numpy())
    # map = ap.mean()
    loss = meters1['loss'].average()
    data_time = meters1['data_time'].average()
    batch_time = meters1['batch_time'].average()

    OP_1, OR_1, OF1_1, CP_1, CR_1, CF1_1 = meters1['ap_meter'].overall()
    OP_k_1, OR_k_1, OF1_k_1, CP_k_1, CR_k_1, CF1_k_1 = meters1['ap_meter'].overall_topk(3)
    logger.info('* Train\nLoss_all: {loss_all:.4f}\nLoss_1: {loss1:.4f}\t mAP: {map:.4f}\t' 
            'Data_time: {data_time:.4f}\t Batch_time: {batch_time:.4f}'.format(
            loss_all=reduced_loss1, loss1=reduced_loss2, map=map1, data_time=data_time, batch_time=batch_time))
    logger.info('OP: {OP:.3f}\t OR: {OR:.3f}\t OF1: {OF1:.3f}\t'
            'CP: {CP:.3f}\t CR: {CR:.3f}\t CF1: {CF1:.3f}'.format(
            OP=OP_1, OR=OR_1, OF1=OF1_1, CP=CP_1, CR=CR_1, CF1=CF1_1))
    logger.info('OP_3: {OP:.3f}\t OR_3: {OR:.3f}\t OF1_3: {OF1:.3f}\t'
            'CP_3: {CP:.3f}\t CR_3: {CR:.3f}\t CF1_3: {CF1:.3f}'.format(
            OP=OP_k_1, OR=OR_k_1, OF1=OF1_k_1, CP=CP_k_1, CR=CR_k_1, CF1=CF1_k_1))

    OP_2, OR_2, OF1_2, CP_2, CR_2, CF1_2 = meters2['ap_meter'].overall()
    OP_k_2, OR_k_2, OF1_k_2, CP_k_2, CR_k_2, CF1_k_2 = meters2['ap_meter'].overall_topk(3)
    logger.info('* Train\nLoss_all: {loss_all:.4f}\nLoss_1: {loss1:.4f}\t mAP: {map:.4f}\t' 
            'Data_time: {data_time:.4f}\t Batch_time: {batch_time:.4f}'.format(
            loss_all=reduced_loss1, loss1=reduced_loss2, map=map2, data_time=data_time, batch_time=batch_time))
    logger.info('OP: {OP:.3f}\t OR: {OR:.3f}\t OF1: {OF1:.3f}\t'
            'CP: {CP:.3f}\t CR: {CR:.3f}\t CF1: {CF1:.3f}'.format(
            OP=OP_2, OR=OR_2, OF1=OF1_2, CP=CP_2, CR=CR_2, CF1=CF1_2))
    logger.info('OP_3: {OP:.3f}\t OR_3: {OR:.3f}\t OF1_3: {OF1:.3f}\t'
            'CP_3: {CP:.3f}\t CR_3: {CR:.3f}\t CF1_3: {CF1:.3f}'.format(
            OP=OP_k_2, OR=OR_k_2, OF1=OF1_k_2, CP=CP_k_2, CR=CR_k_2, CF1=CF1_k_2))
            
    return map1, map2

@torch.no_grad()
def validate(val_loader, model, criterion, epoch, meters1, meters2, device):
    st_time = time.time()
    model.eval()
    Sig = torch.nn.Sigmoid()
    preds_regular1 = []
    targets_1 = []
    preds_regular2 = []
    targets_2 = []
    for i, data in enumerate(val_loader):
        # measure data loading time
        data_time = time.time() - st_time
        meters1['data_time'].update(data_time)

        # inputs, targets, targets_gt, filenames = self.on_start_batch(data)
        inputs = data['image']
        targets = data['target']

        # for voc
        labels = targets.clone()
        targets[targets==0] = 1
        targets[targets==-1] = 0
        inputs = inputs.cuda()
        targets = targets.cuda()

        # outputs = model(inputs, device)
        # loss = criterion(outputs, targets)
        outputs1, outputs2 = model(inputs, device)
        loss1 = criterion(outputs1, targets)
        loss2 = criterion(outputs2, targets)
        loss = loss1+loss2

        batch_time = time.time() - st_time
        meters1['batch_time'].update(batch_time)
        meters1['ap_meter'].add(outputs1.data, labels.data, data['name'])
        meters2['ap_meter'].add(outputs2.data, labels.data, data['name'])
        st_time = time.time()
          
        reduced_loss1 = loss.data
        meters1['loss'].update(to_python_float(reduced_loss1), inputs.size(0))
        reduced_loss2 = loss1.data
        meters2['loss'].update(to_python_float(reduced_loss2), inputs.size(0))

        output_regular1 = Sig(outputs1).cpu()
        preds_regular1.append(output_regular1.cpu().detach())
        targets_1.append(targets.cpu().detach())

        output_regular2 = Sig(outputs2).cpu()
        preds_regular2.append(output_regular2.cpu().detach())
        targets_2.append(targets.cpu().detach())

        if i%args.print_freq == 0:
            logger.info('{}, {} Epoch, {} Iter, Loss_all: {:.4f}, Loss1: {:.4f}, Data time: {:.4f}, Batch time: {:.4f}'.format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  epoch+1, i, 
                    meters1['loss'].value(), meters2['loss'].value(), meters1['data_time'].value(), 
                    meters1['batch_time'].value()))

    map1 = mAP(torch.cat(targets_1).numpy(), torch.cat(preds_regular1).numpy())
    map2 = mAP(torch.cat(targets_2).numpy(), torch.cat(preds_regular2).numpy())

    loss = meters1['loss'].average()
    data_time = meters1['data_time'].average()
    batch_time = meters1['batch_time'].average()

    OP_1, OR_1, OF1_1, CP_1, CR_1, CF1_1 = meters1['ap_meter'].overall()
    OP_k_1, OR_k_1, OF1_k_1, CP_k_1, CR_k_1, CF1_k_1 = meters1['ap_meter'].overall_topk(3)
    logger.info('* Test\nLoss_all: {loss_all:.4f}\nLoss_1: {loss1:.4f}\t mAP: {map:.4f}\t' 
            'Data_time: {data_time:.4f}\t Batch_time: {batch_time:.4f}'.format(
            loss_all=reduced_loss1, loss1=reduced_loss2, map=map1, data_time=data_time, batch_time=batch_time))
    logger.info('OP: {OP:.3f}\t OR: {OR:.3f}\t OF1: {OF1:.3f}\t'
            'CP: {CP:.3f}\t CR: {CR:.3f}\t CF1: {CF1:.3f}'.format(
            OP=OP_1, OR=OR_1, OF1=OF1_1, CP=CP_1, CR=CR_1, CF1=CF1_1))
    logger.info('OP_3: {OP:.3f}\t OR_3: {OR:.3f}\t OF1_3: {OF1:.3f}\t'
            'CP_3: {CP:.3f}\t CR_3: {CR:.3f}\t CF1_3: {CF1:.3f}'.format(
            OP=OP_k_1, OR=OR_k_1, OF1=OF1_k_1, CP=CP_k_1, CR=CR_k_1, CF1=CF1_k_1))

    OP_2, OR_2, OF1_2, CP_2, CR_2, CF1_2 = meters2['ap_meter'].overall()
    OP_k_2, OR_k_2, OF1_k_2, CP_k_2, CR_k_2, CF1_k_2 = meters2['ap_meter'].overall_topk(3)
    logger.info('* Test\nLoss_all: {loss_all:.4f}\nLoss_1: {loss1:.4f}\t mAP: {map:.4f}\t' 
            'Data_time: {data_time:.4f}\t Batch_time: {batch_time:.4f}'.format(
            loss_all=reduced_loss1, loss1=reduced_loss2, map=map2, data_time=data_time, batch_time=batch_time))
    logger.info('OP: {OP:.3f}\t OR: {OR:.3f}\t OF1: {OF1:.3f}\t'
            'CP: {CP:.3f}\t CR: {CR:.3f}\t CF1: {CF1:.3f}'.format(
            OP=OP_2, OR=OR_2, OF1=OF1_2, CP=CP_2, CR=CR_2, CF1=CF1_2))
    logger.info('OP_3: {OP:.3f}\t OR_3: {OR:.3f}\t OF1_3: {OF1:.3f}\t'
            'CP_3: {CP:.3f}\t CR_3: {CR:.3f}\t CF1_3: {CF1:.3f}'.format(
            OP=OP_k_2, OR=OR_k_2, OF1=OF1_k_2, CP=CP_k_2, CR=CR_k_2, CF1=CF1_k_2))
            
    return map1, map2



def save_checkpoint(state, is_best, work_dir = './', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(work_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(work_dir, filename), os.path.join(work_dir, 'model_best.pth.tar'))

def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()

# def adjust_learning_rate(optimizer, epoch, step, len_epoch):
#     """LR schedule that should yield 76% converged accuracy with batch size 256"""
#     factor = epoch // 30

#     if epoch >= 80:
#         factor = factor + 1

#     lr = args.lr*(0.1**factor)


#     # """Warmup"""
#     if epoch < 5:
#         lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     # scheduler.step()

def adjust_learning_rate(optimizer, epoch, epoch_step):
    """ Sets learning rate if it is needed """
    lr_list = []
    decay = 0.1 if sum(epoch == np.array(epoch_step)) > 0 else 1.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay
        lr_list.append(param_group['lr'])

    return np.unique(lr_list)
    

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    # dist.all_reduce(rt, op=dist.reduce_op.SUM)
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()