import numpy as np

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


def train(model, dataloader, validloaders, optimizer, criterion, epoch, val, log=True):
    losses = AverageMeter()
    top1 = AverageMeter()

    for idx, (images, targets) in enumerate(dataloader):
        model.train()

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

        if log:
            if (val and idx % 1 ==0) or (not val and idx == len(dataloader) - 1):
                print('Train: [{0}][{1}/{2}]\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, idx + 1, len(dataloader), loss=losses, top1=top1))
                if val:
                    print('Valid:')
                    for validloader in validloaders:
                        validate(model, validloader, criterion)
        
    return top1


def validate(model, dataloader, criterion, log=True):
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

            if log and idx == len(dataloader) - 1:
                print('Test: [{0}/{1}]\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        idx + 1, len(dataloader), loss=losses, top1=top1))
                
    return top1


def flatten_params(m, numpy_output=True):
    total_params = []
    for param in m.parameters():
            total_params.append(param.view(-1))
    total_params = torch.cat(total_params)
    if numpy_output:
        return total_params.cpu().detach().numpy()
    return total_params

def assign_weights(m, weights):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    with torch.no_grad():
        for param in state_dict.keys():
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
            # print(param, index)
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param] =  nn.Parameter(torch.from_numpy(weights[index:index+param_count].reshape(param_shape)))
            index += param_count
    m.load_state_dict(state_dict)
    return m


def validate_direction(w0, direction, dataloader, dis):
    direction = dis * direction / np.linalg.norm(direction)
    w = w0 + direction

    new_model = Net().cuda()
    new_model = assign_weights(new_model, w)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    top1 = validate(new_model, dataloader, criterion, log=False)
    return top1.avg.item()

def find_boundary(w0, direction, dataloader, lower_dis, upper_dis, boundary_acc, lower_acc=None, upper_acc=None, tol=1e-3):
    if lower_acc is None:
        lower_acc = validate_direction(w0, direction, dataloader, lower_dis)
    if upper_acc is None:
        upper_acc = validate_direction(w0, direction, dataloader, upper_dis)

    # print(lower_dis, lower_acc, upper_dis, upper_acc)

    if abs(upper_dis - lower_dis) < tol:
        return lower_dis
    
    middle_dis = (upper_dis + lower_dis) / 2
    middle_acc = validate_direction(w0, direction, dataloader, middle_dis)
    
    if middle_acc < boundary_acc:
        return find_boundary(w0, direction, dataloader, lower_dis, middle_dis, boundary_acc, lower_acc, middle_acc, tol)
    
    return find_boundary(w0, direction, dataloader, middle_dis, upper_dis, boundary_acc, middle_acc, upper_acc, tol)


def main():
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024,
                                            shuffle=True, num_workers=1)


    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024,
                                            shuffle=False, num_workers=1)

    model1 = Net().cuda()
    model1.load_state_dict(torch.load('outputs/model_bs10_e300.pt', map_location=torch.device('cuda')))

    w0 = flatten_params(model1)
    dataloader_iterator = iter(trainloader)
    boundaries = []
    upper_accs = []
    boundary_acc = 99.9
    upper_dis = 1e5

    for sign in [1.0, -1.0]:
        for i in range(89610):
            direction = np.zeros(89610)
            direction[i] = sign

            upper_acc = 100.0
            it = 0
            while upper_acc > boundary_acc and it < 10:
                try:
                    data = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(trainloader)
                    data = next(dataloader_iterator)

                upper_acc = validate_direction(w0, direction, [data], upper_dis)
                it += 1

            upper_accs.append(upper_acc)
            boundary = find_boundary(w0, direction, [data], 0.0, upper_dis, boundary_acc)
            boundaries.append(boundary)
            print('{}  {} \t {:.3f} \t {:.3f}'.format(sign, i, upper_acc, boundary))

    boundaries = np.array(boundaries)
    upper_accs = np.array(upper_accs)
    np.save('boundaries', boundaries)
    np.save('upper_accs', upper_accs)


if __name__ == '__main__':
    main()
