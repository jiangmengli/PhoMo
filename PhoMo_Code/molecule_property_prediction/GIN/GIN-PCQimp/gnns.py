import torch
from torch.nn import Sequential, Linear, Tanh
from torch_geometric.nn import GINConv, GINEConv
import random


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim, bias=True):
        super(GIN, self).__init__()

        act = Tanh()
        eps = 0.0
        train_eps = False

        self.bn_in = torch.nn.BatchNorm1d(in_channels)

        self.nn1 = Sequential(Linear(in_channels, dim, bias=bias), act)
        self.conv1 = GINConv(self.nn1, eps=eps, train_eps=train_eps)
        self.bn1 = torch.nn.BatchNorm1d(dim)


        self.nn2 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv2 = GINConv(self.nn2, eps=eps, train_eps=train_eps)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.nn3 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv3 = GINConv(self.nn3, eps=eps, train_eps=train_eps)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.nn4 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv4 = GINConv(self.nn4, eps=eps, train_eps=train_eps)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.nn5 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv5 = GINConv(self.nn5, eps=eps, train_eps=train_eps)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, out_channels, bias=False)

    def forward(self, X, X_importance, edge_index):
        x = self.bn_in(X * X_importance)
        xs = [x]

        xs.append(self.conv1(xs[-1], edge_index))
        xs.append(self.conv2(xs[-1], edge_index))
        xs.append(self.conv3(xs[-1], edge_index))
        xs.append(self.conv4(xs[-1], edge_index))
        xs.append(self.conv5(xs[-1], edge_index))
        xs.append(torch.tanh(self.fc1(xs[-1])))

        x = torch.cat(xs[1:], dim=-1)

        return x

class GINE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim, in_channels_e, bias=True):
        super(GINE, self).__init__()

        set_random_seed(10)
        act = Tanh()
        eps = 0.0
        train_eps = False


        self.bn_in = torch.nn.BatchNorm1d(in_channels)
        self.bn_ine = torch.nn.BatchNorm1d(in_channels_e)

        self.nn1 = Sequential(Linear(in_channels, dim, bias=bias), act)
        self.nn1e = Sequential(Linear(in_channels_e, in_channels, bias=bias), act)
        self.conv1 = GINEConv(self.nn1, eps=eps, train_eps=train_eps)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.bn1e = torch.nn.BatchNorm1d(dim)

        self.nn2 = Sequential(Linear(dim, dim, bias=bias), act)
        self.nn2e = Sequential(Linear(in_channels, dim, bias=bias), act)
        self.conv2 = GINEConv(self.nn2, eps=eps, train_eps=train_eps)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.bn2e = torch.nn.BatchNorm1d(dim)

        self.nn3 = Sequential(Linear(dim, dim, bias=bias), act)
        self.nn3e = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv3 = GINEConv(self.nn3, eps=eps, train_eps=train_eps)
        self.bn3 = torch.nn.BatchNorm1d(dim)
        self.bn3e = torch.nn.BatchNorm1d(dim)

        self.nn4 = Sequential(Linear(dim, dim, bias=bias), act)
        self.nn4e = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv4 = GINEConv(self.nn4, eps=eps, train_eps=train_eps)
        self.bn4 = torch.nn.BatchNorm1d(dim)
        self.bn4e = torch.nn.BatchNorm1d(dim)

        self.nn5 = Sequential(Linear(dim, dim, bias=bias), act)
        self.nn5e = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv5 = GINEConv(self.nn5, eps=eps, train_eps=train_eps)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        self.bn5e = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, out_channels, bias=False)

    def forward(self, X, X_importance, edge_index, edge_attr):
        x = self.bn_in(X * X_importance)
        xs = [x]
        edge_attr = self.bn_ine(edge_attr * 1.0)

        edge_attr = self.nn1e(edge_attr)
        xs.append(self.conv1(xs[-1], edge_index, edge_attr))
        edge_attr = self.nn2e(edge_attr)
        xs.append(self.conv2(xs[-1], edge_index, edge_attr))
        edge_attr = self.nn3e(edge_attr)
        xs.append(self.conv3(xs[-1], edge_index, edge_attr))
        edge_attr = self.nn4e(edge_attr)
        xs.append(self.conv4(xs[-1], edge_index, edge_attr))
        edge_attr = self.nn5e(edge_attr)
        xs.append(self.conv5(xs[-1], edge_index, edge_attr))
        xs.append(torch.tanh(self.fc1(xs[-1])))

        x = torch.cat(xs[1:], dim=-1)

        return x


