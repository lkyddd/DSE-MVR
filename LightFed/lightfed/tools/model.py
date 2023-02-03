from collections import OrderedDict
import torch
import copy


def evaluation(model, dataloader, criterion, model_params=None, device=None, eval_full_data=True):
    if model_params is not None:
        model.load_state_dict(model_params)

    if device is not None:
        model.to(device)

    model.eval()
    loss = 0.0
    acc = 0.0
    num = 0
    with torch.no_grad():
        for x, y in dataloader:
            if device is not None:
                x = x.to(device)
                y = y.to(device)
            raw_output = model(x)
            _loss = criterion(raw_output, y)
            _, predicted = torch.max(raw_output, -1)
            _acc = predicted.eq(y).sum()
            _num = y.size(0)
            loss += (_loss * _num).item()
            acc += _acc.item()
            num += _num
            if not eval_full_data:
                break
    loss /= num
    acc /= num
    return loss, acc, num


def get_parameters(model, deepcopy=False):
    ans = OrderedDict()
    for name, params in model.named_parameters():
        if deepcopy:
            params = params.clone().detach()
        ans[name] = params
    return ans


def get_buffers(model, deepcopy=False):
    ans = OrderedDict()
    for name, buffers in model.named_buffers():
        if deepcopy:
            buffers = buffers.clone().detach()
        ans[name] = buffers
    return ans


class CycleDataloader:
    def __init__(self, dataloader, epoch=-1, seed=None) -> None:
        self.dataloader = dataloader
        self.epoch = epoch
        self.seed = seed
        self._data_iter = None
        self._init_data_iter()

    def _init_data_iter(self):
        if self.epoch == 0:
            raise StopIteration()

        if self.seed is not None:
            torch.manual_seed(self.seed + self.epoch)
        self._data_iter = iter(self.dataloader)
        self.epoch -= 1

    def __next__(self):
        try:
            return next(self._data_iter)
        except StopIteration:
            self._init_data_iter()
            return next(self._data_iter)

    def __iter__(self):
        return self
