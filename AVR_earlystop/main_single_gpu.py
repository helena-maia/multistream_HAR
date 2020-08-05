import os
import time
import random
import argparse
import shutil
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import video_transforms
import models
import datasets

from early_stopping.pytorchtools import EarlyStopping

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to datset setting files')
parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
                    choices=["rgb", "flow", "rhythm", "rgb2"],   
                    help='modality: rgb| rgb2 | flow | rhythm')
parser.add_argument('--dataset', '-d', default='ucf101',
                    choices=["ucf101", "hmdb51"],
                    help='dataset: ucf101 | hmdb51')
parser.add_argument('--arch', '-a', metavar='ARCH', default='rgb_resnet152',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: rgb_vgg16)')
parser.add_argument('-s', '--split', default=2, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=25 , type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--iter-size', default=5, type=int,
                    metavar='I', help='iter size as in Caffe to reduce memory usage (default: 5)')
parser.add_argument('--new_length', default=1, type=int,
                    metavar='N', help='length of sampled video frames (default: 1)')
parser.add_argument('--new_width', default=360, type=int,
                    metavar='N', help='resize width (default: 320,360)')
parser.add_argument('--new_height', default=320, type=int,
                    metavar='N', help='resize height (default: 240,320)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[100, 200], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('-pf','--print-freq', default=50,  type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('-sf','--save-freq', default=25, type=int,
                    metavar='N', help='save frequency (default: 25)')
parser.add_argument('--vr_approach', '-vra', default=3, type=int,
                    metavar='N', help='visual rhythm approach (choices: 1 - vertical, 2 - horizontal, 3 - AVR per class, 4 - AVR per video)(default: 3)')
parser.add_argument('--log', metavar='PATH', default='./log', type=str, 
                    help='path to log (default: ./log)')
parser.add_argument('--resume_log', metavar='PATH', default=None, type=str, 
                    help='path to an existing log for non-zero start-epoch (default: None)')
parser.add_argument('-es', action='store_true', 
                    help='Activate early stopping')
parser.add_argument('--pretrain_weights','-pt', default=None, type=str, 
	                help='path to the checkpoint from the pretraining on a different dataset. \
	                (--start-epoch must be 0) (default: None)')


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def createNewDataset(fileNameRead, fileNameWrite, modality_):
    '''
        Generate new training file to train the network with two RGB images,
        taking the first and second images randomly in the first and second 
        half respectively
    '''
    newPathFile = os.path.join(args.settings, args.dataset, fileNameWrite)
    train_setting_file = fileNameRead % (args.split)
    pathFile = os.path.join(args.settings, args.dataset, train_setting_file)    

    file_ = open(pathFile,'r')
    linesFile = file_.readlines()
    detallLines = list()
    for line in linesFile:
        lineInfo = line.split(' ')
        first_frame = random.randint(1, int(lineInfo[1])//2-4)
        second_frame = int(lineInfo[1])//2 + random.randint(1, int(lineInfo[1])//2-4)
        detallLines.append('{} {} {}'.format(lineInfo[0], first_frame, lineInfo[2]))
        detallLines.append('{} {} {}'.format(lineInfo[0], second_frame, lineInfo[2]))
    open(newPathFile,'w').writelines(detallLines)

def main():
    global args, prec_list
    prec_list = []
    args = parser.parse_args()
    full_path = logging(args)

    print(args.modality+" network trained with the split "+str(args.split)+".")

    # create model
    print("Building model ... ")
    exits_model, model = build_model(int(args.start_epoch), args.pretrain_weights)
    if not exits_model:
        return 
    else:
        print("Model %s is loaded. " % (args.arch))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    cudnn.benchmark = True

    # Data transforming
    if args.modality == "rgb" or args.modality == "rgb2":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406] * args.new_length
        clip_std = [0.229, 0.224, 0.225] * args.new_length
    elif args.modality == "flow"  or args.modality == "rhythm":
        is_color = False
        scale_ratios = [1.0, 0.875, 0.75]
        clip_mean = [0.5, 0.5] * args.new_length
        clip_std = [0.226, 0.226] * args.new_length
    else:
        print("No such modality. Only rgb and flow supported.")

    new_size= 299 if args.arch.find("inception_v3")>0 else 224

    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)
    train_transform = video_transforms.Compose([
            #video_transforms.Scale((256)),
            video_transforms.MultiScaleCrop((new_size, new_size), scale_ratios),
            video_transforms.RandomHorizontalFlip(),
            video_transforms.ToTensor(),
            normalize,
        ])

    if args.es:
        val_transform = video_transforms.Compose([
                # video_transforms.Scale((256)),
                video_transforms.CenterCrop((new_size)),
                video_transforms.ToTensor(),
                normalize,
            ])
    
    modality_ = "rgb" if (args.modality == "rhythm" or args.modality[:3] == "rgb") else "flow"
 
    if args.modality == "rgb2":
        createNewDataset("train_split%d.txt" , "new_train.txt",modality_)
        #createNewDataset("val_%s_split%d.txt", "new_val.txt",modality_)

    # data loading  
    train_setting_file = "new_train.txt" if args.modality == "rgb2" else "train_split%d.txt" % (args.split)
    train_split_file = os.path.join(args.settings, args.dataset, train_setting_file)

    if not os.path.exists(train_split_file):# or not os.path.exists(val_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (args.settings))

    extension = ".png" if args.dataset == "hmdb51" and args.modality == "rhythm" else ".jpg"
    direction_file = "direction.txt" if args.vr_approach == 3 else "direction_video.txt"
    direction_path = os.path.join(args.settings, args.dataset, direction_file)

    train_dataset = datasets.__dict__['dataset'](root=args.data,
                                                  source=train_split_file,
                                                  phase="train",
                                                  modality=args.modality,
                                                  is_color=is_color,
                                                  new_length=args.new_length,
                                                  new_width=args.new_width,
                                                  new_height=args.new_height,
                                                  video_transform=train_transform,
                                                  approach_VR = args.vr_approach,
                                                  extension = extension,
                                                  direction_path = direction_path)



    if args.es:
        val_setting_file = "val_split%d.txt" % (args.split) 
        val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)
        
        if not os.path.exists(val_split_file):
            print("No split file exists in %s directory. Preprocess the dataset first" % (args.settings))

        val_dataset = datasets.__dict__['dataset'](root=args.data,
                                                      source=val_split_file,
                                                      phase="val",
                                                      modality=args.modality,
                                                      is_color=is_color,
                                                      new_length=args.new_length,
                                                      new_width=args.new_width,
                                                      new_height=args.new_height,
                                                      video_transform=val_transform,
                                                      approach_VR = args.vr_approach,
                                                      extension = extension, 
                                                      direction_path = direction_path)
    
        print('{} samples found, {} train samples and {} validation samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))
    else:
        print('{} train samples found.'.format(len(train_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if args.es:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        
        early_stop = EarlyStopping(verbose=True, 
                                   log_path=os.path.join(full_path, "early_stopping.json"))

    is_best = False

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        if args.es:
            # evaluate on validation set
            losses = validate(val_loader, model, criterion)

            is_best = early_stop(losses.avg, epoch)

        if (epoch + 1) % args.save_freq == 0 or is_best:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint_"+args.modality+"_split_"+str(args.split)+".pth.tar")
            es_val = float('inf') if not args.es else early_stop.val_loss_min
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'val_loss_min': es_val
            }, is_best, checkpoint_name, os.path.join(full_path,"checkpoints"))

        prec_name =  "%03d_%s" % (epoch + 1, "prec_split_"+str(args.split)+".txt")
        save_precision(prec_name, os.path.join(full_path,"precision"))

        if args.es and early_stop.early_stop:
            break

    if not args.es: # Final model
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint_"+args.modality+"_split_"+str(args.split)+".pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'val_loss_min': float('inf')
            }, True, checkpoint_name, os.path.join(full_path,"checkpoints"))


def logging(args):
    args_dict = vars(args)
    args_dict['hostname'] = os.uname()[1]
    if 'VIRTUAL_ENV' in os.environ.keys(): args_dict['virtual_env'] = os.environ['VIRTUAL_ENV']
    else: 
        print("WARNING: No virtualenv activated")
        args_dict['virtual_env'] = None
   
    if args.start_epoch != 0 and args.resume_log: #resume
        with open(os.path.join(args.resume_log,'args.json'), 'r') as json_file:
            args_dict2 = json.load(json_file)
            if args_dict != args_dict2: 
            	print("WARNING: args differ")
            	diff_items1 = [ (k,args_dict[k],args_dict2[k]) for k in args_dict if k in args_dict2 and args_dict[k] != args_dict2[k]]
            	if(diff_items1): print("Different items: ", diff_items1)
            	diff_items2 = [ k for k in args_dict if k not in args_dict2 ]
            	diff_items3 = [ k for k in args_dict2 if k not in args_dict ]
            	if(diff_items2 or diff_items3): print("Missing items: ", diff_items2, diff_items3)
        return args.resume_log
    else: #new training
        timestamp = time.time() 
 
        full_path = os.path.join(args.log,str(timestamp))
        os.makedirs(full_path)

        with open(os.path.join(full_path,'args.json'), 'w') as json_file:
            json.dump(args_dict, json_file)

        os.system('pip freeze > '+os.path.join(full_path,'requirements.txt'))

        os.makedirs(os.path.join(full_path,"checkpoints"))
        os.makedirs(os.path.join(full_path,"precision"))

        print("Saving everything to directory %s." % (full_path))

        return full_path

    
def build_model(resume_epoch, pretrain_path):
    is_new = (resume_epoch==0)
    found = True
    num_classes = 51 if args.dataset =='hmdb51' else 101
    num_channels = 1 if args.modality == 'rhythm' else 20 if args.modality=='flow' else 3
    model = models.__dict__[args.arch](pretrained=is_new, channels=num_channels, num_classes=num_classes)
    if not is_new:
        path = os.path.join(args.resume_log,'checkpoints', '{0:03d}_checkpoint_{1}_split_{2}.pth.tar'.format(resume_epoch,args.modality,args.split))
        print(path)
        if os.path.isfile(path):    
            print('loading checkpoint {0:03d} ...'.format(resume_epoch))    
            params = torch.load(path)
            model.load_state_dict(params['state_dict'])
            print('loaded checkpoint {0:03d}'.format(resume_epoch))
        else:
            print('ERROR: No checkpoint found')
            found = False
    elif pretrain_path:
        print(pretrain_path)
        if os.path.isfile(pretrain_path):
            print('Transfer learning: loading checkpoint')
            params = torch.load(pretrain_path)
            print("Epoch: %d, Prec: %.2f"%(params['epoch'],params['best_prec1']))
            model_dict = model.state_dict()
            fc_state_dict = {k: v for k, v in model_dict.items() if 'fc_action' in k}
            state_dict = {**params['state_dict'], **fc_state_dict} #replace fc_action layers to keep old values 
            model.load_state_dict(state_dict)
            print('loaded checkpoint')
        else:
            print('ERROR: No checkpoint found')
            found = False

    model.cuda()
    return found, model

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch = 0.0
    acc_mini_batch = 0.0

    for i, (input, target) in enumerate(train_loader):
        input = input.float().cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        outputs= model(input_var)

        loss = None
        # for nets that have multiple outputs such as inception
        if isinstance(outputs, tuple):
            loss = sum((criterion(o,target_var) for o in outputs))
            outputs_data = outputs[0].data
        else:
            loss = criterion(outputs, target_var)
            outputs_data = outputs.data    

        # measure accuracy and record loss
        prec1, prec3 = accuracy(outputs_data, target, topk=(1, 3))
        acc_mini_batch += prec1.item()
        loss = loss / args.iter_size
        loss_mini_batch += loss.data.item()
        loss.backward()

        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()

            # losses.update(loss_mini_batch/args.iter_size, input.size(0))
            # top1.update(acc_mini_batch/args.iter_size, input.size(0))
            losses.update(loss_mini_batch, input.size(0))
            top1.update(acc_mini_batch/args.iter_size, input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            loss_mini_batch = 0
            acc_mini_batch = 0

            if (i+1) % args.print_freq == 0:

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, i+1, len(train_loader)+1, batch_time=batch_time, loss=losses, top1=top1))
                prec_list.append('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(   
                       epoch, i+1, len(train_loader)+1, batch_time=batch_time, loss=losses, top1=top1))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.float().cuda(async=True)
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top3.update(prec3.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top3=top3))
                prec_list.append('Validation: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(   
                   i, len(val_loader), batch_time=batch_time, loss=losses, 
                   top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))

    return losses


def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    best_path = os.path.join(resume_path, 'model_best_'+args.modality+'_split_'+str(args.split)+'.pth.tar')
    torch.save(state, cur_path)

    if is_best:
        shutil.copyfile(cur_path, best_path)


def save_precision(prec_filename, resume_path):
    global prec_list
    prec_path = os.path.join(resume_path, prec_filename)
    np.savetxt(prec_path, prec_list, fmt="%s", delimiter="\n")
    prec_list = []


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""

    decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    lr = args.lr * decay
    print("Current learning rate is %4.6f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()

