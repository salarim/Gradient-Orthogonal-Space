import numpy as np
from scipy.linalg import null_space

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(model, dataloader, memloader, validloaders, optimizer, criterion, epoch, val):
    losses = AverageMeter()
    top1 = AverageMeter()

    if memloader is not None:
        memloader_iterator = iter(memloader)

    for idx, (images, targets) in enumerate(dataloader):
        model.train()
        if memloader is not None:
            try:
                mem_images, mem_targets = next(memloader_iterator)
            except StopIteration:
                memloader_iterator = iter(memloader)
                mem_images, mem_targets = next(memloader_iterator)
            mem_bsz = mem_images.shape[0]
            new_bsz = images.shape[0]
            images = torch.cat([images, mem_images], dim=0)
            targets = torch.cat([targets, mem_targets], dim=0)

        bsz = targets.shape[0]
        images, targets = images.cuda(), targets.cuda()

        output = model(images)
        loss = criterion(output, targets)
  
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (val and idx % 1 ==0) or (not val and idx == len(dataloader) - 1):
            print('Train: [{0}][{1}/{2}]\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, idx + 1, len(dataloader), loss=losses, top1=top1))
            if val:
                print('Valid:')
                for validloader in validloaders:
                    validate(model, validloader, criterion)


def validate(model, dataloader, criterion):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            bsz = targets.shape[0]
            images, targets = images.cuda(), targets.cuda()

            output = model(images)
            loss = criterion(output, targets)

            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            top1.update(acc1[0], bsz)

            if idx == len(dataloader) - 1:
                print('Test: [{0}/{1}]\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        idx + 1, len(dataloader), loss=losses, top1=top1))


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                        shuffle=True, num_workers=1)
    trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=10,
                                        shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                         shuffle=False, num_workers=1)

    EPOCHS = 300
    VAL_FREQ = 1

    model = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                            lr=0.001,
                            momentum=0.9,
                            weight_decay=1e-4)
    model, criterion = model.cuda(), criterion.cuda()

    for epoch in range(1, EPOCHS+1):
        train(model, trainloader, None, testloader, optimizer, criterion, epoch, False)
        if epoch % VAL_FREQ == 0 or epoch == EPOCHS:
            validate(model, testloader, criterion)

    print('#'*80)


    grads = None
    shapes = []

    for idx, (images, targets) in enumerate(trainloader2):
        images, targets = images.cuda(), targets.cuda()
        output = model(images)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()

        flat_grads = None
        for param in model.parameters():
            if idx == 0:
                shapes.append(list(param.grad.shape))
            flat_grad = param.grad.detach().view(-1).cpu().numpy()
            if flat_grads is None:
                flat_grads = flat_grad
            else:
                flat_grads = np.append(flat_grads, flat_grad, axis=0)
        
        flat_grads = np.expand_dims(flat_grads, axis=0)
        if grads is None:
            grads = flat_grads
        else:
            grads = np.append(grads, flat_grads, axis=0)

    print(shapes)
    print(grads.shape)

    ns = null_space(grads)
    print(ns.shape)

    torch.save(model.state_dict(), 'model.pt')
    np.save('null_space', ns)


if __name__ == '__main__':
    main()
