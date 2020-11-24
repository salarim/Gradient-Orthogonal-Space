from copy import deepcopy

import numpy as np
import torch
import torchvision
from torchvision import transforms

from main import Net, AverageMeter, accuracy, validate


def main():

    shapes = [[100, 784], [100], [100, 100], [100], [10, 100], [10]]
    nb_params = 89610
    step_sizes = [10.0, 3.0, 1.0]

    null_spaces = np.load('outputs/null_space_bs10.npy').T

    model = Net()
    model.load_state_dict(torch.load('outputs/model_bs10.pt', map_location=torch.device('cpu')))
    criterion = torch.nn.CrossEntropyLoss()
    model, criterion = model.cuda(), criterion.cuda()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000,
                                            shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                            shuffle=False, num_workers=1)


    for j, null_space in enumerate(null_spaces):
        print('### Null Space index:', j)
        null_space = null_space / np.linalg.norm(null_space)

        for step_size in step_sizes:
            print('Step Size:', step_size)

            model2= deepcopy(model)
            idx = 0
            cum_idx = 0
            for i, (name, param) in enumerate(model2.named_parameters()):
                shape = shapes[i]
                size = np.prod(shape)

                delta = null_space[cum_idx:cum_idx+size].reshape(shape)
                delta = torch.tensor(delta).cuda()

                new_param = param.data + step_size * delta
                param.data.copy_(new_param)
                cum_idx += size
            
            validate(model2, trainloader, criterion)
            # validate(model2, testloader, criterion)
            print()


if __name__ == "__main__":
    main()
