import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse

parser = argparse.ArgumentParser(description='Generate all posible maxcut configuration and return their optimized parameters')

parser.add_argument("--number_of_nodes", type=int)
parser.add_argument("--train_csv", type=str)
parser.add_argument("--test_csv", type=str)
parser.add_argument("--dataset_size", type=int, default=5000)

args = parser.parse_args()

class Net(nn.Module):

    def __init__(self, bit_len, p):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(bit_len, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2*p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def list_of_binaries_from_decimal(num,Length_list):
    """take a number and return list of its binary in a list.
        add zeros to reach similar length
    """
    num_string = bin(num)[2:]
    if len(num_string)>=Length_list:
        return [int(char) for char in num_string]
    else:
        Add_Zeros_Num = Length_list - len(num_string)
        num_string = Add_Zeros_Num*'0' + num_string
        return [int(char) for char in num_string]

def df_to_dataloader(df, p, n, batch_size=32, train=True):
    # print(df.head())
    edges = ['x%s' % i for i in range(n)]
    X = torch.tensor(df[edges].values)
    # X = torch.tensor([list_of_binaries_from_decimal(int(z), n) for z in X])
    if train:
        params = ['b%s'%i for i in range(p)] + ['g%s'%i for i in range(p)]
        y = torch.tensor(df[params].values)
    else:
        y = torch.zeros(X.shape[0])
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    return dataloader

epochs = 100
bit_len = int(args.number_of_nodes*(args.number_of_nodes-1)/2)
train_df = pd.read_csv("train_datasets/"+args.train_csv+".csv", nrows=args.dataset_size)
p = int((train_df.shape[1]-2-bit_len)/2)
test_df = pd.read_csv("test_datasets/"+args.test_csv+".csv")

# msk = np.random.rand(len(df)) < 0.8
# train_df = df[msk]
# test_df = df[~msk]

train_loader, test_loader = df_to_dataloader(train_df, p, bit_len), df_to_dataloader(test_df, p, bit_len, train=False)
print('train size: %d. Test size: %d' %(len(train_df),len(test_df)))

net = Net(bit_len, p)
criterion = nn.MSELoss()
optimaizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(epochs):
    net.train()
    running_loss = 0
    n=0
    for X,y in train_loader:
        optimaizer.zero_grad()
        output = net(X.float())
        loss = criterion(output, y.float())
        loss.backward()
        optimaizer.step()
        n += X.shape[0]
        running_loss += loss.item()
    
    # net.eval()
    # n_val = 0
    # running_val_loss = 0
    # with torch.no_grad():
    #     for X,y in test_loader:
    #         output = net(X.float())
    #         loss = criterion(output, y.float())
    #         n_val += X.shape[0]
    #         running_val_loss += loss.item()
    # print("Epoch: %d loss: %0.4f, val_loss:%0.4f" %(epoch, running_loss/n, running_val_loss/n_val))
    print("Epoch: %d loss: %0.4f" % (epoch, running_loss / n))

net.eval()
graphs, preds = [], []
with torch.no_grad():
    for X, y in test_loader:
        output = net(X.float())
        graphs.append((X.detach().cpu().numpy()))
        preds.append(output.detach().cpu().numpy())
graphs = np.concatenate(graphs)
graphs_df = pd.DataFrame(graphs)

preds = np.concatenate(preds)
params = ['b%s'%i for i in range(p)] + ['g%s'%i for i in range(p)]
preds_df = pd.DataFrame(preds, columns=params)

result = pd.concat([graphs_df, preds_df], axis=1, join="inner")
result.to_csv("preds/train_%s_test_%s_preds_dataset_size_%s.csv" %(args.train_csv, args.test_csv, args.dataset_size), index=False)