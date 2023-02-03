import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
# from transformers import BertModel


def model_pull(args):

    if args.model_type == 'LR':
        return LR(args)
    # ---------------------------------------

    elif args.model_type == 'TinyNN':
        return TinyNN(args)

    elif args.model_type == 'LeNet':
        return LeNet(args)

    elif args.model_type == 'ResNet_20':
        return ResNet(BasicBlock, [3, 3, 3], args)

    elif args.model_type == 'ResNet_32':
        return ResNet(BasicBlock, [5, 5, 5], args)

    elif args.model_type == 'VGG_11':
        return VGGNet(args)

    elif args.model_type == 'Model_CIFAR10':
        return Model4CIFAR10()

    elif args.model_type == 'Model_CIFAR100':
        return Model4CIFAR100()

    elif args.model_type == 'Model_FEMNIST':
        return Model4FEMNIST()

    elif args.model_type == 'Model_MNIST':
        return Model4MNIST()
    else:
        raise Exception(f"unkonw model_type: {args.model_type}")
    # elif args.model_type == 'Bert':
    #     self.model = Bert(args)
    # ---------------------------------------


'''LR'''
class LR(torch.nn.Module):
    def __init__(self, args):
        super(LR, self).__init__()
        self.args = args

        class_num = args.data_distributer.class_num
        feature_size = np.prod(args.data_distributer.x_shape)

        self.fc = nn.Linear(in_features=feature_size,
                            out_features=class_num, bias=True)

    def forward(self, x):
        if self.args.data_set in ['MNIST', 'CIFAR-10', 'CIFAR-100']:
            x = x.view(x.shape[0], -1).to(torch.float32)
        elif self.args.data_set in ['COVERTYPE', 'A9A', 'W8A']:
            x = x.to(torch.float32)
        x = self.fc(x)
        return x


'''TinyNN'''
class TinyNN(nn.Module):
    def __init__(self, args):
        super(TinyNN, self).__init__()
        self.args = args

        class_num = args.data_distributer.class_num
        feature_size = np.prod(args.data_distributer.x_shape)

        self.fc1 = nn.Linear(in_features=feature_size, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=200)
        self.fc3 = nn.Linear(in_features=200, out_features=class_num)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.args.data_set in ['MNIST', 'CIFAR-10', 'CIFAR-100']:
            x = x.view(x.shape[0], -1).to(torch.float32)
        elif self.args.data_set in ['COVERTYPE', 'A9A', 'W8A']:
            x = x.to(torch.float32)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


'''LeNet '''
class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        self.args = args

        class_num = args.data_distributer.class_num
        channel_size = args.data_distributer.x_shape[0]

        self.conv1 = nn.Conv2d(channel_size, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        if self.args.data_set in ['CIFAR-10', 'CIFAR-100']:
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
        elif self.args.data_set in ['MNIST']:
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, class_num)

    def forward(self, x):
        x = x.to(torch.float32)
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


'''ResNet-20/32'''
def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        x = x.to(torch.float32)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args):
        super(ResNet, self).__init__()
        self.args = args

        class_num = args.data_distributer.class_num
        channel_size = args.data_distributer.x_shape[0]

        self.in_planes = 16
        self.conv1 = nn.Conv2d(channel_size, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        if self.args.data_set in ['CIFAR-10', 'CIFAR-100']:
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, option='A')
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, option='A')
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, option='A')
        else:
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, option='B')
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, option='B')
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, option='B')
        self.linear = nn.Linear(64, class_num)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, option):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(torch.float32)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


'''Bert'''
# class Bert(nn.Module):
#     def __init__(self, args):
#         super(Bert, self).__init__()
#         self.args = args
#         class_num, _, _ = args.data_distributer.get_data_shape()
#         self.bert = BertModel.from_pretrained('bert-base-cased')
#         self.dropout = nn.Dropout(0.5)
#         self.linear = nn.Linear(768, class_num)
#         self.relu = nn.ReLU()
#
#     def forward(self, input_id, mask):
#         _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
#         dropout_output = self.dropout(pooled_output)
#         linear_output = self.linear(dropout_output)
#         final_layer = self.relu(linear_output)
#         return final_layer


'''VGG-11'''
class VGGNet(nn.Module):
    def __init__(self, args):
        super(VGGNet, self).__init__()
        self.args = args
        class_num = args.data_distributer.class_num

        self.feature = self.vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512)))
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, class_num)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def vgg_block(self, num_convs, in_channels, out_channels):
        net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True)]
        for i in range(num_convs - 1):
            net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            net.append(nn.ReLU(True))
        net.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*net)

    def vgg_stack(self, num_convs, channels):
        net = []
        for n, c in zip(num_convs, channels):
            in_c = c[0]
            out_c = c[1]
            net.append(self.vgg_block(n, in_c, out_c))
        return nn.Sequential(*net)


'''Model_CIFAR10'''
class Model4CIFAR10(nn.Module):
    def __init__(self):
        super(Model4CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(5 * 5 * 64, 512)
        self.drop_out1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 128)
        self.drop_out2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # 100x3x32x32
        x = F.relu(self.conv1(x), inplace=True)
        # 100x32x28x28
        x = F.max_pool2d(x, 2, 2)
        # 100x32x14x14
        x = F.relu(self.conv2(x), inplace=True)
        # 100x64x10x10
        x = F.max_pool2d(x, 2, 2)
        # 100x64x5x5
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc1(x), inplace=True)
        x = self.drop_out1(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.drop_out2(x)
        x = self.fc3(x)
        return x


'''Model_CIFAR100'''
class Model4CIFAR100(nn.Module):
    def __init__(self):
        super(Model4CIFAR100, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(5 * 5 * 64, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 100)

    def forward(self, x):
        # 50x3072 -> 50x3x32x32
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, 3, 32, 32)).float() / 255.
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 64)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x


'''Model_FEMNIST'''
class Model4FEMNIST(nn.Module):
    def __init__(self):
        super(Model4FEMNIST, self).__init__()
        self.con = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 62)
        )

    def forward(self, x):
        x = self.con(x)
        x = x.flatten(start_dim=1)
        output = self.fc(x)
        return output


'''Moodel_MNIST'''
class Model4MNIST(nn.Module):
    def __init__(self):
        super(Model4MNIST, self).__init__()
        self.cov1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.cov2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 1x28x28
        x = F.relu(self.cov1(x), inplace=True)
        # 32x24x24
        x = F.max_pool2d(x, 2, 2)
        # 32x12x12
        x = F.relu(self.cov2(x), inplace=True)
        # 64x8x8
        x = F.max_pool2d(x, 2, 2)
        # 64x4x4
        x = x.flatten(start_dim=1)
        # [1024]
        x = F.relu(self.fc1(x), inplace=True)
        # 512
        x = self.fc2(x)
        # 10
        return x
