# meta_sgd_module.py

import torch
import torch.nn as nn
import copy

class MetaSGDModule(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.lrs = nn.ParameterDict()
        self.name_map = {}

        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                safe_name = name.replace('.', '_')
                self.lrs[safe_name] = nn.Parameter(torch.ones_like(param) * 0.01)
                self.name_map[name] = safe_name

        self.fisher = {}  # param_name → tensor
        self.prev_params = {}  # snapshot θ*

    def forward(self, *args, return_sample_features=False, **kwargs):
        return self.base_model(*args, return_sample_features=return_sample_features, **kwargs)

    def compute_updated_params(self, loss, scale=1.0, create_graph=False):
        grads = torch.autograd.grad(loss, self.base_model.parameters(), create_graph=create_graph, retain_graph=True, allow_unused=True)
        updated_params = {}
        for (name, param), grad in zip(self.base_model.named_parameters(), grads):
            if param.requires_grad:
                safe_name = self.name_map[name]
                if grad is not None:
                    updated_params[name] = param - self.lrs[safe_name] * scale * grad
                else:
                    updated_params[name] = param
        return updated_params

    def functional_forward(self, x, updated_params, return_sample_features=False):
        model_copy = copy.deepcopy(self.base_model)
        model_dict = model_copy.state_dict()
        new_dict = {}

        for name, param in model_dict.items():
            if name in updated_params:
                new_dict[name] = updated_params[name]
            else:
                new_dict[name] = param

        model_copy.load_state_dict(new_dict)
        model_copy.eval()
        return model_copy(x, return_sample_features=return_sample_features)

    def clone_params(self):
        """保存当前 θ∗ 快照"""
        self.prev_params = {
            name: p.detach().clone() for name, p in self.base_model.named_parameters() if p.requires_grad
        }

    def compute_fisher(self, dataloader, feature_extractor, device, num_tasks=20):
        """计算 Fisher 信息矩阵"""
        fisher = {name: torch.zeros_like(p) for name, p in self.base_model.named_parameters() if p.requires_grad}
        self.base_model.eval()

        loader = iter(dataloader)
        for _ in range(num_tasks):
            s_imgs, s_labels, _, _, _ = next(loader)
            s_imgs = s_imgs.expand(-1, 3, 84, 84).to(device)
            with torch.no_grad():
                feats = feature_extractor(s_imgs)

            output = self.base_model(feats)
            loss = output.norm()
            loss.backward()

            for name, p in self.base_model.named_parameters():
                if p.grad is not None and p.requires_grad:
                    fisher[name] += p.grad.data.clone().pow(2)

        for name in fisher:
            fisher[name] /= num_tasks
        self.fisher = fisher

    def penalty(self):
        """计算 EWC 正则项 loss: L_EWC = (λ/2) * Σ_i F_ii * (θ_i - θ_i*)^2"""
        loss = 0.0
        for name, param in self.base_model.named_parameters():
            if name in self.fisher and name in self.prev_params:
                fisher_val = self.fisher[name]  # F_ii (Fisher矩阵对角元素)
                prev_val = self.prev_params[name]  # θ_i*
                # 根据论文公式：L_EWC = (λ/2) * Σ_i F_ii * (θ_i - θ_i*)^2
                loss += 0.5 * (fisher_val * (param - prev_val).pow(2)).sum()
        return loss
