import logging
from collections import OrderedDict

import torch
from experiments.models.model import model_pull
from lightfed.tools.funcs import set_seed
from lightfed.tools.model import CycleDataloader, get_parameters
from torch import nn


class ClientTrainer:
    def __init__(self, args, client_id):
        self.args = args
        self.client_id = client_id
        self.device = args.device
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay

        self.train_dataloader = args.data_distributer.get_client_train_dataloader(client_id)
        self.train_batch_data_iter = CycleDataloader(self.train_dataloader)
        self.test_dataloader = args.data_distributer.get_client_test_dataloader(client_id)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        set_seed(args.seed + 657)
        self.model = model_pull(args).to(self.device)
        _params = get_parameters(self.model, deepcopy=True)
        _zeros = self._zero_like(_params)
        self.model_params = _params
        self.model_params_mid = _params

        self.grad_t = None

    def _zero_like(self, params):
        ans = OrderedDict()
        for name, weight in params.items():
            ans[name] = torch.zeros_like(weight, device=self.device).detach()
        return ans

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
        logging.debug(f"train_locally_step for step: {step}")
        batch_x, batch_y = self._new_random_batch()
        self.grad_t = self._get_grad_(self.model_params, batch_x, batch_y)

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
