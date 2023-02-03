import logging
from collections import OrderedDict

import torch
from experiments.models.model import model_pull
from lightfed.tools.aggregator import ModelStateAvgAgg
from lightfed.tools.funcs import formula, set_seed
from lightfed.tools.model import CycleDataloader, get_parameters
from torch import nn
from torch.utils.data.dataloader import DataLoader


class ClientTrainer:
    def __init__(self, args, client_id):
        self.args = args
        self.client_id = client_id
        self.device = args.device
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        self.alpha = None  # 会在外面设置

        self.train_dataloader = args.data_distributer.get_client_train_dataloader(client_id)
        self.train_batch_data_iter = CycleDataloader(self.train_dataloader)
        self._train_big_batch_dataloader = DataLoader(dataset=self.train_dataloader.dataset, batch_size=2048, shuffle=False)  # 大批次用于快速计算全局梯度

        self.test_dataloader = args.data_distributer.get_client_test_dataloader(client_id)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        set_seed(args.seed + 657)
        self.model = model_pull(args).to(self.device)
        _params = get_parameters(self.model, deepcopy=True)
        _zeros = self._zero_like(_params)
        self.model_params = _params
        self.model_params_old = _params
        self.model_params_tau_old = _params
        self.model_params_mid = _params

        self.grad_y = _zeros
        self.grad_y_tau_old = _zeros

        self.grad_h = _zeros
        self.grad_h_tau_old = _zeros

        self.grad_v = None
        self.reset_grad_v()

    def reset_grad_v(self):
        self.grad_v = self._grad_of_full_data(self.model_params)

    def _zero_like(self, params):
        ans = OrderedDict()
        for name, weight in params.items():
            ans[name] = torch.zeros_like(weight, device=self.device).detach()
        return ans

    def _grad_of_full_data(self, params):
        _grad_agg = ModelStateAvgAgg()
        n = len(self._train_big_batch_dataloader.dataset)
        for x, y in self._train_big_batch_dataloader:
            _weight = len(y) / n
            _grad = self._get_grad_(params, x, y)
            _grad_agg.put(_grad, _weight)
        return _grad_agg.get_and_clear()

    def _get_grad_(self, params, x, y):
        self.model.load_state_dict(params, strict=False)
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        x = x.to(self.device)
        y = y.to(self.device)

        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        loss.backward()

        grad = OrderedDict()
        with torch.no_grad():
            for name, weight in self.model.named_parameters():
                _g = weight.grad.detach()
                if 'bias' not in name:
                    _g += (weight * self.weight_decay).detach()
                grad[name] = _g

        self.model.zero_grad(set_to_none=True)
        return grad

    def train_locally_step(self, step):
        """算法的第12行到第15行
        """
        logging.debug(f"train_locally_step for step: {step}")
        self.model_params = self.model_params_mid
        batch_x, batch_y = self._new_random_batch()
        g_t_1 = self._get_grad_(self.model_params, batch_x, batch_y)
        g_t = self._get_grad_(self.model_params_old, batch_x, batch_y)
        self.grad_v = formula(lambda g_t_1, grad_v, g_t: g_t_1 + (1 - self.alpha) * (grad_v - g_t),
                              g_t_1, self.grad_v, g_t)
        self.model_params_old = self.model_params

    def _new_random_batch(self):
        x, y = next(self.train_batch_data_iter)
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y

    def get_eval_info(self, step):
        res = {'step': step, 'client_id': self.client_id}

        # loss, acc, num = evaluation(model=self.model,
        #                             dataloader=self.train_dataloader,
        #                             criterion=self.criterion,
        #                             model_params=self.model_params,
        #                             device=self.device)
        # res.update(train_loss=loss, train_acc=acc, train_sample_size=num)

        # loss, acc, num = evaluation(model=self.model,
        #                             dataloader=self.test_dataloader,
        #                             criterion=self.criterion,
        #                             model_params=self.model_params,
        #                             device=self.device)
        # res.update(test_loss=loss, test_acc=acc, test_sample_size=num)

        return res
