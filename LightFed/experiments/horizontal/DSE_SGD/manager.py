import logging
import math
import os

import numpy as np
import pandas as pd
import torch
from experiments.models.model import model_pull
from lightfed.core import BaseServer
from lightfed.tools.aggregator import ModelStateAvgAgg
from lightfed.tools.funcs import (consistent_hash, formula, model_size,
                                  save_pkl, set_seed)
from lightfed.tools.model import evaluation, get_buffers
from torch import nn

from trainer import ClientTrainer


class ServerManager(BaseServer):
    def __init__(self, ct, args):
        super().__init__(ct)
        self.args = args
        self.device = args.device
        self.client_num = args.client_num

        self.full_train_dataloader = self.args.data_distributer.get_train_dataloader()
        self.full_test_dataloader = self.args.data_distributer.get_test_dataloader()
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        set_seed(args.seed + 657)
        self.model = model_pull(args).to(self.device)
        self._initial_model_buffer = get_buffers(self.model, deepcopy=True)
        self._model_with_batchnorm = bool(len(self._initial_model_buffer))
        logging.info(f"model_with_batchnorm: {self._model_with_batchnorm}")

        self.local_sample_numbers = [len(args.data_distributer.get_client_train_dataloader(client_id).dataset)
                                     for client_id in range(args.client_num)]
        self.client_params_list = [None] * self.client_num
        self.global_params = None
        self.global_params_aggregator = ModelStateAvgAgg()
        self.consensus_degree_list = []
        self.accum_communication_cost = np.zeros(shape=(self.client_num, self.client_num))
        self.accum_communication_cost_list = []
        self.client_eval_info = []
        self.global_eval_info = []

        self.comm_round = args.comm_round
        self.unfinished_client_num = -1

        self.step = -1

    def start(self):
        logging.info("start...")
        self.next_step()

    def end(self):
        logging.info("end...")

        super_params = self.args.__dict__.copy()
        del super_params['data_distributer']
        del super_params['weight_matrix']
        del super_params['log_level']
        super_params['device'] = super_params['device'].type

        ff = f"{self.args.app_name}-{consistent_hash(super_params, code_len=64)}.pkl"
        logging.info(f"output to {ff}")

        result = {'super_params': super_params,
                  'global_eval_info': pd.DataFrame(self.global_eval_info),
                  'client_eval_info': pd.DataFrame(self.client_eval_info),
                  'consensus_degree': pd.DataFrame(self.consensus_degree_list),
                  'communication_cost': self.accum_communication_cost_list}
        save_pkl(result, f"{os.path.dirname(__file__)}/Result/{ff}")

        self._ct_.shutdown_cluster()

    def end_condition(self):
        return self.step > self.comm_round - 1

    def next_step(self):
        self.step += 1
        self.unfinished_client_num = self.client_num
        self.global_params_aggregator.clear()

        for client_id in range(self.client_num):
            self.client_params_list[client_id] = None

            self._ct_.get_node('client', client_id) \
                .fed_client_train_step(self.step)

        if ((self.step + 1) % self.args.tau) == 0:
            self._ct_.barrier()
            self._ct_.barrier()

    def fed_finish_client_train_step(self,
                                     step,
                                     client_id,
                                     client_model_params,
                                     communication_cost,
                                     eval_info):
        logging.debug(f"train step of client_id:{client_id} step:{step} was finished")
        assert self.step == step

        self.client_params_list[client_id] = client_model_params
        weight = self.local_sample_numbers[client_id]
        self.global_params_aggregator.put(client_model_params, weight)

        self.accum_communication_cost += communication_cost
        self.client_eval_info.append(eval_info)

        self.unfinished_client_num -= 1
        if not self.unfinished_client_num:
            self.global_params = self.global_params_aggregator.get_and_clear()
            self._set_consensus_degree()
            self._set_communication_cost()
            self._set_global_eval_info()

            logging.debug(f"train step:{step} is finished")
            self.next_step()

    def _set_consensus_degree(self):
        consensus_degree = 0.0
        for client_params in self.client_params_list:
            consensus_dict = formula(lambda global_p, client_p: ((global_p - client_p) ** 2).sum().detach(),
                                     self.global_params,
                                     client_params)
            consensus_degree += sum(consensus_dict.values())
        consensus_degree = torch.sqrt(consensus_degree) / self.args.client_num
        self.consensus_degree_list.append({'step': self.step, 'consensus_degree': consensus_degree.item()})

    def _set_communication_cost(self):
        self.accum_communication_cost_list.append({'step': self.step, 'accum_communication_cost': self.accum_communication_cost.copy()})

    def _set_global_eval_info(self):
        if self.step % self.args.eval_step_interval:
            logging.info(f"skip eval step: {self.step}")
            return

        self._load_and_reset_model_bn_buffer()

        eval_info = {'step': self.step}
        loss, acc, num = evaluation(model=self.model,
                                    dataloader=self.full_train_dataloader,
                                    criterion=self.criterion,
                                    device=self.device,
                                    eval_full_data=False)
        eval_info.update(train_loss=loss, train_acc=acc, train_sample_size=num)

        loss, acc, num = evaluation(model=self.model,
                                    dataloader=self.full_test_dataloader,
                                    criterion=self.criterion,
                                    device=self.device,
                                    eval_full_data=self.args.eval_on_full_test_data)
        eval_info.update(test_loss=loss, test_acc=acc, test_sample_size=num)

        logging.info(f"global eval info:{eval_info}")
        self.global_eval_info.append(eval_info)

    def _load_and_reset_model_bn_buffer(self):
        self.model.load_state_dict(self.global_params, strict=False)
        if self._model_with_batchnorm:
            self.model.train()
            with torch.no_grad():
                for x, _ in self.full_train_dataloader:
                    self.model(x.to(self.device))


class ClientManager(BaseServer):
    def __init__(self, ct, args):
        super().__init__(ct)
        self.args = args
        self.device = args.device
        self.gamma_list = args.gamma_list.copy()
        self.gamma_list.append((math.inf, None))
        self.gamma = 0.0
        self.next_gamma_stage = -1

        self.weight_matrix = args.weight_matrix
        self.client_id = self._ct_.role_index

        self.communication_cost = np.zeros(shape=(args.client_num, args.client_num))

        self.trainer = ClientTrainer(args, self.client_id)
        self.nei_params_aggregator = ModelStateAvgAgg()
        self.nei_grad_y_aggregator = ModelStateAvgAgg()
        self.unreceived_neighbor_num = 0
        self.step = 0

    def start(self):
        logging.info("start...")

    def end(self):
        logging.info("end...")

    def end_condition(self):
        return False

    def clear(self):
        self.communication_cost[:, :] = 0.0
        self.nei_params_aggregator.clear()
        self.nei_grad_y_aggregator.clear()

    def fed_client_train_step(self, step):
        self.step = step
        self._set_gamma(step)
        self.clear()
        logging.debug(f"training client_id:{self.client_id}, step:{step}")

        # 算法第5行
        self.trainer.model_params_mid = formula(lambda p_t, g_t: p_t - self.gamma * g_t,
                                                self.trainer.model_params, self.trainer.grad_g_t)

        if ((step + 1) % self.args.tau) == 0:
            self.trainer.grad_h = formula(lambda tau_old, mid: tau_old - mid,
                                          self.trainer.model_params_tau_old,
                                          self.trainer.model_params_mid)
            section_of_grad_y = formula(lambda y_tau, h, h_tau: y_tau + h - h_tau,
                                        self.trainer.grad_y_tau_old,
                                        self.trainer.grad_h,
                                        self.trainer.grad_h_tau_old)
            self._ct_.barrier()
            nei_client_id_list = np.flatnonzero(self.weight_matrix[self.client_id, :])
            self.nei_grad_y_aggregator.clear()
            self.unreceived_neighbor_num = len(nei_client_id_list)
            for nei_client_id in nei_client_id_list:
                if nei_client_id == self.client_id:
                    self.fed_receive_neighbor_section_of_grad_y(self.client_id, section_of_grad_y)
                else:
                    self._ct_.get_node("client", nei_client_id) \
                        .set(deepcopy=False) \
                        .fed_receive_neighbor_section_of_grad_y(self.client_id, section_of_grad_y)
                    self.communication_cost[self.client_id, nei_client_id] += model_size(section_of_grad_y)

        else:
            self.trainer.model_params = self.trainer.model_params_mid
            self.trainer.train_locally_step(step)
            self.finish_train_step()

    def fed_receive_neighbor_section_of_grad_y(self, nei_client_id, section_of_grad_y):
        logging.debug(f"fed_receive_neighbor_section_of_grad_y, nei_client_id:{nei_client_id}")

        weight = self.weight_matrix[nei_client_id, self.client_id]
        self.nei_grad_y_aggregator.put(section_of_grad_y, weight)
        self.unreceived_neighbor_num -= 1

        if not self.unreceived_neighbor_num:
            logging.debug(f"all neighbor_section_of_grad_y was received client:{self.client_id}, step:{self.step}")
            self.trainer.grad_y = self.nei_grad_y_aggregator.get_and_clear()

            self.trainer.grad_y_tau_old = self.trainer.grad_y
            self.trainer.grad_h_tau_old = self.trainer.grad_h

            section_of_params = formula(lambda params_tau, grad_y: params_tau - grad_y,
                                        self.trainer.model_params_tau_old,
                                        self.trainer.grad_y)
            self._ct_.barrier()
            nei_client_id_list = np.flatnonzero(self.weight_matrix[self.client_id, :])
            self.nei_params_aggregator.clear()
            self.unreceived_neighbor_num = len(nei_client_id_list)
            for nei_client_id in nei_client_id_list:
                if nei_client_id == self.client_id:
                    self.fed_receive_neighbor_section_of_params(self.client_id, section_of_params)
                else:
                    self._ct_.get_node("client", nei_client_id) \
                        .set(deepcopy=False) \
                        .fed_receive_neighbor_section_of_params(self.client_id, section_of_params)
                    self.communication_cost[self.client_id, nei_client_id] += model_size(section_of_params)

    def fed_receive_neighbor_section_of_params(self, nei_client_id, section_of_params):
        logging.debug(f"fed_receive_neighbor_section_of_params, nei_client_id:{nei_client_id}")

        weight = self.weight_matrix[nei_client_id, self.client_id]
        self.nei_params_aggregator.put(section_of_params, weight)
        self.unreceived_neighbor_num -= 1

        if not self.unreceived_neighbor_num:
            logging.debug(f"all neighbor_section_of_params was received client:{self.client_id}, step:{self.step}")
            _agg_params = self.nei_params_aggregator.get_and_clear()
            self.trainer.model_params = _agg_params
            self.trainer.model_params_tau_old = _agg_params
            self.trainer.train_locally_step(self.step)

            self.finish_train_step()

    def _set_gamma(self, step):
        if step >= self.next_gamma_stage:
            _, self.gamma = self.gamma_list.pop(0)
            self.next_gamma_stage = self.gamma_list[0][0]
            logging.info(f"update gamma to :{self.gamma}, next_gamma_stage:{self.next_gamma_stage}")

    def finish_train_step(self):
        eval_info = self.trainer.get_eval_info(self.step)
        logging.debug(f"finish_train_step step:{self.step}, client_id:{self.client_id}")

        self._ct_.get_node("server") \
            .set(deepcopy=False) \
            .fed_finish_client_train_step(self.step,
                                          self.client_id,
                                          self.trainer.model_params,
                                          self.communication_cost,
                                          eval_info)
