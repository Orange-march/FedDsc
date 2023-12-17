import numpy as np
import torch
import torchvision.transforms
from torchvision import datasets
from torchvision import datasets, transforms
from non_iid_dataset import MyDataset
from util import TwoCropTransform


class Arguments:
    def __init__(self) -> None:
        self.N_CLIENTS = 10
        self.batch_size = 128
        self.test_batch_size = 1000
        self.DIRICHLET_ALPHA = 0.1
        self.data_path = 1
        self.user_data_distr = []
        self.user_data_cata = []

#cifar10
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
normalize = transforms.Normalize(mean=mean, std=std)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])


def dirichlet_split_noniid(train_labels, n_classes, alpha, n_clients):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    '''
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标

    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


def class_count(n_classes, client, train_labels):
    ls = []
    ls_count = []
    ls_cls = []
    for icls in range(n_classes):
        count = 0
        ls_class = []
        for idx in client:
            if icls == train_labels[idx]:
                count += 1
                ls_class.append(idx)
        if count > 0:
            ls.append(ls_class)
            ls_count.append(count)
        ls_cls.append(icls)

    return ls, ls_count, ls_cls


def get_public_dataset(pb_idx, train_data,args):
    pb_train_data = []
    pb_train_label = []

    for i in range(len(pb_idx)):
        for idx in pb_idx[i]:
            pb_train_data.append(train_data.data[idx])
            pb_train_label.append(train_data.targets[idx])

    pb_train = [pb_train_data, pb_train_label]

    pb_train_dataset = MyDataset(pb_train, train=True,
                                 transform=train_transform
                                 )

    pb_train_loader = torch.utils.data.DataLoader(pb_train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)

    return pb_train_loader


def remove_pb(pb_idx, datas, train_labels):
    datas = np.delete(datas, pb_idx)

    train_labels = np.delete(train_labels, pb_idx)

    return datas, train_labels


def divide_dataset(args, client_idcs, n_classes, train_labels, train_data):
    clients_train_loaders = []

    for user_idx in range(args.N_CLIENTS):

        ls, ls_count, ls_cls = class_count(n_classes, client_idcs[user_idx], train_labels)

        # print(ls_count)

        args.user_data_distr.append(ls_count)
        args.user_data_cata.append(ls_cls)

        client_train_data = []
        client_train_label = []

        for i in range(len(ls)):
            for idx in ls[i]:
                client_train_data.append(train_data[idx])
                client_train_label.append(train_labels[idx])

        client_train = [client_train_data, client_train_label]

        client_train_dataset = MyDataset(client_train, train=True,
                                         transform=TwoCropTransform(train_transform)
                                         )

        train_loader = torch.utils.data.DataLoader(client_train_dataset, shuffle=True, batch_size=args.batch_size,
                                                   drop_last=True)

        clients_train_loaders.append(train_loader)

    return clients_train_loaders


def get_user_dataset():
    np.random.seed(42)

    args = Arguments()

    train_data = datasets.CIFAR10(root=args.data_path, download=True, train=True, )

    test_data = datasets.CIFAR10(root=args.data_path, download=True, train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914,0.4822,0.4465),(0.2471,0.2435,0.2616))#网上训练好的参数
            ]
        )
    )

    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=args.test_batch_size)

    train_labels = np.array(train_data.targets)

    n_classes = train_labels.max() + 1

    public_idx = dirichlet_split_noniid(train_labels, n_classes, 1000000, 10)
    pb_ls = [public_idx[0], public_idx[1], public_idx[2], public_idx[3][0:2000]]
    pb_train_loaders = get_public_dataset(pb_ls, train_data,args)

    #train_labels, datas = remove_pb(public_idx[1], train_data, train_labels)

    client_idcs = dirichlet_split_noniid(train_labels, n_classes, alpha=args.DIRICHLET_ALPHA, n_clients=args.N_CLIENTS)

    clients_train_loaders = divide_dataset(args, client_idcs, n_classes, train_labels, train_data.data)

    return  pb_train_loaders, clients_train_loaders, test_loader

# pb, _, _ = get_user_dataset()
