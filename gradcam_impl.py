from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from my_typings import T


class _BaseWrapper(object):
    def __init__(self, model: T._net_t):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot.clone()

    def forward(self, image: Tensor):
        self.image_shape = image.shape[2:]
        self.logits: Tensor = self.model(image).requires_grad_().clone()
        self.probs = F.softmax(self.logits, dim=1).clone()
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids: Tensor):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids).clone()
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)  # type: ignore
        del one_hot

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image: Tensor):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad
        self.image.grad.zero_()

        gradient_cp = gradient.cpu()
        del gradient
        return gradient_cp


class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model: T._net_t):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_in[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))  # type: ignore


class Deconvnet(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model: T._net_t):
        super(Deconvnet, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients and ignore ReLU
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_out[0]),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))  # type: ignore


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model: T._net_t, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool: Dict[str, Tensor] = {}
        self.grad_pool: Dict[str, Tensor] = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input: Tensor, output: Tensor):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(
                module,
                grad_in: Union[Tensor, Tuple[Tensor, ...]],
                grad_out: Union[Tensor, Tuple[Tensor, ...]],
            ):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool: Dict[str, Tensor], target_layer: str):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer: str):
        fmaps = self._find(self.fmap_pool, target_layer).clone()
        grads = self._find(self.grad_pool, target_layer).clone()
        weights = F.adaptive_avg_pool2d(grads, (1, 1))

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam: Tensor = F.interpolate(gcam, self.image_shape, mode="bilinear", align_corners=False)

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        gcam_cp = gcam.cpu()
        del gcam, fmaps, grads, weights
        return gcam_cp
