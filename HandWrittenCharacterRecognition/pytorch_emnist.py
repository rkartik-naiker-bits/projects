# Simple CNN for the MNIST Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from Constants import Constants

batch_size = 200
num_inputs = 784
num_outputs=47

params_cnn_1 = Constants.params_cnn_1
params_cnn_2 = Constants.params_cnn_2
params_cnn_3 = Constants.params_cnn_3

params="1"
dropout_flag=False

class cnn_model_1(nn.Module):
    def __init__(self):
        super(cnn_model_1, self).__init__()
        lst = params_cnn_1[params]
        a = lst[0]
        b = lst[1]
        c = lst[2]
        d = lst[3]
        self.conv1 = nn.Conv2d(1, a, (5, 5))
        self.conv2 = nn.Conv2d(a, b, (5, 5))
        self.fc1 = nn.Linear(b * 4 * 4, c)
        if d!=0:
            self.fc2 = nn.Linear(c, d)
            next_in=d
        else:
            next_in=c
        self.fc3 = nn.Linear(next_in, num_outputs)

    def forward(self, x):
        lst = params_cnn_1[params]
        b=lst[1]
        d = lst[3]
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if d!=0:
            x=F.relu(self.fc2(x))
        if dropout_flag:
            x=F.dropout(x,0.2)
        x = F.log_softmax(self.fc3(x),dim=1)
        return x

class cnn_model_2(nn.Module):

    def __init__(self):
        super(cnn_model_2, self).__init__()
        lst = params_cnn_2[params]
        a = lst[0]
        b = lst[1]
        c = lst[2]
        self.conv1 = nn.Conv2d(1, a, (5, 5))
        self.fc1 = nn.Linear(a * 12 * 12, b)
        if c!=0:
            self.fc2 = nn.Linear(b, c)
            next_in=c
        else:
            next_in=b
        self.fc3 = nn.Linear(next_in, num_outputs)

    def forward(self, x):
        lst = params_cnn_2[params]
        a=lst[0]
        c = lst[2]
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = F.relu(self.fc1(x))
        if c!=0:
            x=F.relu(self.fc2(x))
        if dropout_flag:
            x=F.dropout(x,0.2)
        x = F.log_softmax(self.fc3(x),dim=1)
        return x

class cnn_model_3(nn.Module):

    def __init__(self):
        super(cnn_model_3, self).__init__()
        lst = params_cnn_3[params]
        a = lst[0]
        b = lst[1]
        c = lst[2]
        d = lst[3]
        e = lst[4]
        f = lst[5]

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=a, kernel_size=3, stride=1)
        self.batchNorm1 = nn.BatchNorm2d(num_features=a)
        self.conv2 = nn.Conv2d(in_channels=a, out_channels=b, kernel_size=3, stride=1)
        self.batchNorm2 = nn.BatchNorm2d(num_features=b)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=b, out_channels=c, kernel_size=3, stride=1)
        self.batchNorm3 = nn.BatchNorm2d(num_features=c)
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=d, kernel_size=3, stride=1)
        self.batchNorm4 = nn.BatchNorm2d(num_features=d)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        if e != 0:
            self.fc0 = nn.Linear(in_features=1024, out_features=e)
            next_in = e
        else:
            next_in = 1024
        self.fc1 = nn.Linear(in_features=next_in, out_features=f)
        self.batchNorm5 = nn.BatchNorm1d(num_features=f)
        if dropout_flag:
            self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=f, out_features=num_outputs)

    def forward(self, x):
        lst = params_cnn_3[params]
        e = lst[4]
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = self.max_pool_1(x)
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
        x = self.max_pool_2(x)
        x = x.view(x.size(0), -1)
        if e!=0:
            x=self.fc0(x)
        x = F.relu(self.batchNorm5(self.fc1(x)))
        if(dropout_flag):
            x = self.dropout1(x)
        x = F.log_softmax(self.fc2(x))
        return x


class Main():
    def train(self,model, device, train_loader, optimizer, epoch, batch_size):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.reshape(batch_size, 1, 28, 28)
            optimizer.zero_grad()
            output = model(data.float())
            loss = F.nll_loss(output, target.long())
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    def test(self,model, device, test_loader, batch_size):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.reshape(batch_size, 1, 28, 28)
                output = model(data.float())
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def load_data(self,file):
        data = pd.read_csv(file, header=None)
        X = data.iloc[:, 1:]
        Y = data.iloc[:, 0]
        X = X.to_numpy()
        Y = Y.to_numpy()
        data = []
        for i in range(0, len(X)):
            data.append([X[i], Y[i]])
        return data

    def transform_data(self,data):
        data = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        return data

    def main(self):

        train_data = self.transform_data(self.load_data(Constants.DATA_SET_PATH + 'emnist-balanced-train.csv'))
        test_data = self.transform_data(self.load_data(Constants.DATA_SET_PATH + 'emnist-balanced-test.csv'))

        torch.manual_seed(1)
        device = torch.device("cpu")
        cnn_model=cnn_model_3()
        #cnn_model = cnn_model_2()
        #cnn_model = cnn_model_3()
        model = cnn_model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        for epoch in range(0, 10):
            self.train(model, device, train_data, optimizer, epoch, batch_size)
            self.test(model, device, train_data, batch_size)
            self.test(model, device, test_data, batch_size)

if __name__ == '__main__':
    Main().main()


