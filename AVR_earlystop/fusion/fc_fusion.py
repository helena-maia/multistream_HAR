import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import math
import time
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "../")
from early_stopping.pytorchtools import EarlyStopping

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
#torch.backends.cudnn.deterministic = False

class Fusion(nn.Module):
    def __init__(self, arq=[('L',10,101),('R'),('D',0.9)]):
        super(Fusion, self).__init__()

        modules = []

        for l in arq:
            if l[0]  == 'L':
                modules.append(nn.Linear(l[1],l[2]))
            elif l[0] == 'R':
                modules.append(nn.ReLU(True))
            elif l[0] == 'D':
                modules.append(nn.Dropout(p=l[1]))

        self.fc = nn.Sequential(*modules)
        self._initialize_weights()

    
    def forward(self, x):
        x = self.fc(x)
        return x
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class Dataset(data.Dataset):
    def __init__(self,dataset,classes):
        self.dataset = dataset
        self.classes = classes

    def __getitem__(self,idx):
        data = torch.from_numpy(self.dataset[idx])
        target = self.classes[idx]
        return data,target

    def __len__(self):
        return len(self.dataset)

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


def accuracy(output, target, topk=(1,), validate=False):
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

def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        input = input.float().cuda(async=True)
        target = target.cuda(async=True)

        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec2, prec3 = accuracy(output.data, target, topk=(1,2,3), validate=True)
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top2.update(prec2.item(), input.size(0))
        top3.update(prec3.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@2 {top2.val:.3f} ({top2.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top2=top2,top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@2 {top2.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top2=top2, top3=top3))

    return losses, top1.avg

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""

    decay = 0.1 ** (sum(epoch >= np.array([50,100,150,200,250])))
    _lr = lr * decay
    print("Current learning rate is %4.6f:" % _lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = _lr

def fc_fusion(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts, arq = [('L',303,101)], n_epochs = 100, lr=0.1, batch_size=1000):
    scaler = StandardScaler()
    X_tr_ = scaler.fit_transform(X_tr)
    X_vl_ = scaler.transform(X_vl)
    X_ts_ = scaler.transform(X_ts)

    print_freq = 1 #impressao dos batches
    save_freq = 5 #num de epocas para validar

    model = Fusion(arq)
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=lr,
                                momentum=0.9,
                                weight_decay=5e-4)

    cudnn.benchmark = True

    train_dataset = Dataset(X_tr_, y_tr)
    val_dataset = Dataset(X_vl_, y_vl)
    test_dataset = Dataset(X_ts_, y_ts)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    early_stop = EarlyStopping(verbose=True, 
                               log_path=os.path.join(full_path, "early_stopping.json"))

    best_model = None

    for epoch in range(n_epochs):
        adjust_learning_rate(optimizer, epoch, lr)

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        model.train()

        end = time.time()
        optimizer.zero_grad()

        #each batch
        for i, (input, target) in enumerate(train_loader):
            input = input.float().cuda(async=True)
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)

            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            acc_mini_batch = prec1.item()

            loss = criterion(output, target_var)
            loss_mini_batch = loss.data.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.update(loss_mini_batch, input.size(0))
            top1.update(acc_mini_batch, input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % print_freq == 0:

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i+1, len(train_loader)+1, batch_time=batch_time, loss=losses, top1=top1))

        #validate
        if (epoch + 1) % save_freq == 0:
            losses, _ = validate(val_loader, model, criterion, print_freq)
            is_best = early_stop(losses.avg, epoch)

            if is_best:
                best_model = model.state_dict()

            if early_stop.early_stop:
                break

    model.load_state_dict(best_model)
    _, prec = validate(test_loader, model, criterion, print_freq)

    return prec