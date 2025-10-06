### Untouched original script from https://github.com/pytorch/examples/blob/main/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
import os
import logging
import random
import numpy as np

# logging.basicConfig(filename = f"/home/inesgp/cnn_training/slurm/randomseed_v2_mnist_train_{os.environ['TASK_ID']}.log", filemode='w', level=logging.INFO,)
# logging.basicConfig(filename = f"/home/inesgp/cnn_training/slurm/ieee_mnist_train_{os.environ['TASK_ID']}.log", filemode='w', level=logging.INFO,)
# logging.basicConfig(filename = f"{os.environ['LOG_DIR']}/slurm/fuzzy_mnist_train_sr_{os.environ['TASK_ID']}.log", filemode='w', level=logging.INFO,)
logging.basicConfig(filename = f"{os.environ['LOG_DIR']}/weight_init_{os.environ['INIT_MODE']}.log", filemode='w', level=logging.INFO,)



class Net_weights(nn.Module):
    def __init__(self, init_mode='xavier_uniform'):
        super(Net_weights, self).__init__()
        self.init_mode = init_mode

        self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=True)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if self.init_mode == 'uniform':
                    nn.init.uniform_(m.weight)
                    nn.init.uniform_(m.bias)
                elif self.init_mode == 'normal':
                    nn.init.normal_(m.weight)
                    nn.init.normal_(m.bias)
                elif self.init_mode == 'ones':
                    nn.init.ones_(m.weight)
                    nn.init.ones_(m.bias)
                elif self.init_mode == 'zeros':
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                elif self.init_mode == 'identity':
                    if isinstance(m, nn.Linear):
                        nn.init.eye_(m.weight)
                # elif self.init_mode == 'eye':
                #     if isinstance(m, nn.Linear):
                #         nn.init.eye_(m.weight)
                elif self.init_mode == 'dirac':
                    if isinstance(m, nn.Conv2d):
                        nn.init.dirac_(m.weight)
                elif self.init_mode == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif self.init_mode == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif self.init_mode == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.init_mode == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.init_mode == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                elif self.init_mode == 'sparse':
                    if isinstance(m, nn.Linear):
                        nn.init.sparse_(m.weight, sparsity=0.1)
                
                # if m.bias is not None:
                #     nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=True)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



def train(args, model, device, train_loader, optimizer, epoch):

    model.train()
    correct = 0
    total = 0
    loss = 0

    def closure():
        nonlocal correct
        nonlocal total
        nonlocal loss
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        loss = F.nll_loss(output, target)
        loss.backward()
        return loss

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if os.environ['INIT_MODE'] == 'LBFGS': 
            optimizer.step(closure)
        else:
            optimizer.zero_grad()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            loss = F.nll_loss(output, target)
            # loss.requires_grad_()
            loss.backward()
            optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.17e}%)]\tAccuracy: {}\tLoss: {:.17e}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                100. * correct / total, 
                loss.item()))
            if args.dry_run:
                break

        # break


def test(model, device, test_loader, val_flag='Test'):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logging.info('\n{} set: Average loss: {:.17e}, Accuracy: {}/{} ({:.17e}%)\n'.format(
        val_flag,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--init-mode', type=str, default=None, 
                        help='set weight initialization mode (default: None)')
    parser.add_argument('--optimizer-mode', type=str, default=None, 
                        help='set optimizer (default: None)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    # dataset1 = datasets.MNIST('./data', train=True, download=True,
    #                    transform=transform)
    full_train_dataset = datasets.MNIST(f'./data', train=True, download=False, transform=transform)
    val_ratio = 0.1  # 10% for validation
    train_size = int((1 - val_ratio) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))

    dataset2 = datasets.MNIST(f'./data', train=False,
                       transform=transform)
    generator = torch.Generator()
    generator.manual_seed(0)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs, generator=generator,)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs, generator=generator,)  # use test_kwargs for val_loader too
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs, generator=generator,)

    if args.init_mode:
        model = Net_weights(init_mode=args.init_mode).to(device)
    else:
        model = Net().to(device)

    if args.optimizer_mode == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer_mode == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer_mode == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer_mode == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=args.lr)
    elif args.optimizer_mode == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    logging.info('Logging started')
    
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, val_loader, val_flag='Validation')
        scheduler.step()
        if args.save_model:
            # torch.save(model.state_dict(), f"/home/inesgp/cnn_training/random_seeds/randomseed_v2_mnist_cnn_{os.environ['TASK_ID']}.pt")
            # torch.save(model.state_dict(), f"/home/inesgp/cnn_training/ieee/mnist_cnn_{os.environ['TASK_ID']}.pt")
            torch.save(model.state_dict(), f"{os.environ['LOG_DIR']}/mnist_cnn_{os.environ['TASK_ID']}.pt")

    test(model, device, test_loader)
    logging.shutdown()

    if args.save_model:
        # torch.save(model.state_dict(), f"/home/inesgp/cnn_training/random_seeds/randomseed_v2_mnist_cnn_{os.environ['TASK_ID']}.pt")
        # torch.save(model.state_dict(), f"/home/inesgp/cnn_training/ieee/mnist_cnn_{os.environ['TASK_ID']}.pt")
        torch.save(model.state_dict(), f"{os.environ['LOG_DIR']}/mnist_cnn_{os.environ['TASK_ID']}.pt")


if __name__ == '__main__':
    main()

