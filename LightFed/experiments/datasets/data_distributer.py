import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets as vision_datasets


class TransDataset(Dataset):
    def __init__(self, dataset, transform) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return self.transform(img), label

    def __len__(self):
        return len(self.dataset)


class ListDataset(Dataset):
    def __init__(self, data_list) -> None:
        super().__init__()
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


class DataDistributer:
    def __init__(self, args, dataset_dir=None, cache_dir=None):
        if dataset_dir is None:
            dataset_dir = os.path.abspath(os.path.join(__file__, "../../../../dataset"))

        if cache_dir is None:
            cache_dir = f"{dataset_dir}/cache_data"

        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.args = args
        self.client_num = args.client_num
        self.batch_size = args.batch_size

        self.class_num = None
        self.x_shape = None
        self.client_train_dataloaders = []
        self.client_test_dataloaders = []
        self.train_dataloaders = None
        self.test_dataloaders = None


        _dataset_load_func = getattr(self, f'_load_{args.data_set.replace("-","_")}')
        _dataset_load_func()

    def get_client_train_dataloader(self, client_id):
        return self.client_train_dataloaders[client_id]

    def get_client_test_dataloader(self, client_id):
        return self.client_test_dataloaders[client_id]

    def get_train_dataloader(self):
        return self.train_dataloaders

    def get_test_dataloader(self):
        return self.test_dataloaders

    def _load_MNIST(self):
        self.class_num = 10
        self.x_shape = (1, 28, 28)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = vision_datasets.MNIST(root=f"{self.dataset_dir}/MNIST", train=True, download=True, transform=transform)
        test_dataset = vision_datasets.MNIST(root=f"{self.dataset_dir}/MNIST", train=False, download=True, transform=transform)

        self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size, shuffle=True)
        self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size, shuffle=True)

        client_train_datasets, client_test_datasets = self._split_dataset(train_dataset, test_dataset)
        self.client_train_dataloaders = []
        self.client_test_dataloaders = []
        for client_id in range(self.client_num):
            _train_dataset = client_train_datasets[client_id]
            _test_dataset = client_test_datasets[client_id]
            _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, shuffle=True)
            _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=self.args.batch_size, shuffle=True)
            self.client_train_dataloaders.append(_train_dataloader)
            self.client_test_dataloaders.append(_test_dataloader)

    def _load_CIFAR_10(self):
        self.class_num = 10
        self.x_shape = (3, 32, 32)

        train_transform = transforms.Compose([
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

        raw_train_dataset = vision_datasets.CIFAR10(root=f"{self.dataset_dir}/CIFAR-10", train=True, download=True)
        raw_test_dataset = vision_datasets.CIFAR10(root=f"{self.dataset_dir}/CIFAR-10", train=False, download=True)

        train_dataset = TransDataset(raw_train_dataset, train_transform)
        test_dataset = TransDataset(raw_test_dataset, test_transform)
        self.train_dataloaders = DataLoader(dataset=train_dataset, batch_size=self.args.eval_batch_size, shuffle=True)
        self.test_dataloaders = DataLoader(dataset=test_dataset, batch_size=self.args.eval_batch_size, shuffle=True)

        raw_client_train_datasets, raw_client_test_datasets = self._split_dataset(raw_train_dataset, raw_test_dataset)
        self.client_train_dataloaders = []
        self.client_test_dataloaders = []
        for client_id in range(self.client_num):
            _raw_train_dataset = raw_client_train_datasets[client_id]
            _raw_test_dataset = raw_client_test_datasets[client_id]
            _train_dataset = TransDataset(_raw_train_dataset, train_transform)
            _test_dataset = TransDataset(_raw_test_dataset, test_transform)
            _train_dataloader = DataLoader(dataset=_train_dataset, batch_size=self.args.batch_size, shuffle=True)
            _test_dataloader = DataLoader(dataset=_test_dataset, batch_size=self.args.batch_size, shuffle=True)
            self.client_train_dataloaders.append(_train_dataloader)
            self.client_test_dataloaders.append(_test_dataloader)

    def _load_CIFAR_100(self):
        raise Exception("not implement yet")
        _transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.class_num = 100
        self.x_shape = (3, 32, 32)

        _train_dataset = vision_datasets.CIFAR100(root=f"{self.dataset_dir}/CIFAR-100", train=True, download=True, transform=_transform)
        self.train_dataloaders = self._to_tensor_dataset(_train_dataset)

        _test_dataset = vision_datasets.CIFAR100(root=f"{self.dataset_dir}/CIFAR-100", train=False, download=True, transform=_transform)
        self.test_dataloaders = self._to_tensor_dataset(_test_dataset)

    def _split_dataset(self, train_dataset, test_dataset):
        if self.args.data_partition_mode == 'iid':
            partition_proportions = np.full(shape=(self.class_num, self.client_num), fill_value=1/self.client_num)
        elif self.args.data_partition_mode == 'non_iid_dirichlet':
            partition_proportions = np.random.dirichlet(alpha=np.full(shape=self.client_num, fill_value=self.args.non_iid_alpha), size=self.class_num)
        else:
            raise Exception(f"unknow data_partition_mode:{self.args.data_partition_mode}")

        client_train_datasets = self._split_dataset_by_proportion(train_dataset, partition_proportions)
        client_test_datasets = self._split_dataset_by_proportion(test_dataset, partition_proportions)
        return client_train_datasets, client_test_datasets

    def _split_dataset_by_proportion(self, dataset, partition_proportions):
        grouped_data = [[] for _ in range(self.class_num)]
        for item in dataset:
            grouped_data[item[1]].append(item)

        client_data_list = [[] for _ in range(self.client_num)]

        for client_id in range(self.client_num):
            class_id = np.random.randint(low=0, high=self.class_num)
            client_data_list[client_id].append(grouped_data[class_id].pop())

        for class_id in range(self.class_num):
            data_list = grouped_data[class_id]
            data_num = len(data_list)

            selected_client_id = np.random.choice(a=self.client_num, size=data_num, p=partition_proportions[class_id])
            for item, client_id in zip(data_list, selected_client_id):
                client_data_list[client_id].append(item)

        client_datasets = []
        for client_data in client_data_list:
            np.random.shuffle(client_data)
            _dataset = ListDataset(client_data)
            client_datasets.append(_dataset)

        return client_datasets


if __name__ == "__main__":
    import argparse

    def get_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--batch_size', type=int, default=128)

        parser.add_argument('--data_set', type=str, default='CIFAR-10')

        parser.add_argument('--data_partition_mode', type=str, default='non_iid_dirichlet',
                            choices=['iid', 'non_iid_dirichlet'])

        parser.add_argument('--non_iid_alpha', type=float, default=0.5)

        parser.add_argument('--client_num', type=int, default=10)

        parser.add_argument('--device', type=torch.device, default='cuda')

        parser.add_argument('--seed', type=int, default=0)

        parser.add_argument('--app_name', type=str, default='DecentLaM')

        args = parser.parse_args(args=[])

        return args

    args = get_args()
    dd = DataDistributer(args)

