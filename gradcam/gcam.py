import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.cm as mpl_cm
import numpy as np
import torch
import torch.nn as nn
from modules import T
from PIL import Image
from torch import Tensor
from torchvision import transforms


@dataclass
class Extractor:
    model: T._net
    target_layer: str
    gradients: Optional[Tensor] = None

    def save_grad(self, grad):
        self.gradients = grad

    def forward_pass_on_conv(self, x: Tensor):
        conv_output = None
        layer: str
        module: nn.Module
        for layer, module in self.model.features._modules.items():  # type: ignore
            x = module(x)
            if layer == self.target_layer:
                x.register_hook(self.save_grad)
                conv_output = x
        return conv_output, x

    def forward_pass(self, x: Tensor):
        conv_output, x = self.forward_pass_on_conv(x)
        x = x.view(x.size()[0], -1)
        x = self.model.classifier(x)  # type: ignore
        return conv_output, x


@dataclass
class GradCAM:
    model: T._net
    target_layer: str
    input_size: Tuple[int, int] = (60, 60)

    extractor: Extractor = field(init=False)

    def __post_init__(self) -> None:
        self.model.eval()
        self.extractor = Extractor(self.model, self.target_layer)

    def generate_cam(self, image: Tensor, device: torch.device, target_cls: int = None):
        conv_output, model_output = self.extractor.forward_pass(image)
        if target_cls is None:
            target_cls = np.argmax(model_output.detach().cpu().numpy())

        one_hot = torch.zeros(size=(1, model_output.size()[-1])).to(device)  # type: ignore
        one_hot[0][target_cls] = 1

        self.model.features.zero_grad()  # type: ignore
        self.model.classifier.zero_grad()  # type: ignore

        model_output.backward(gradient=one_hot, retain_graph=True)  # type: ignore
        guided_gradients = self.extractor.gradients.detach().cpu().numpy()[0]
        target = conv_output.detach().cpu().numpy()[0]
        weights = np.mean(guided_gradients, axis=(1, 2))

        cam = np.ones(target.shape[1:], dtype=np.float)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)  # type: ignore
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)  # type: ignore
        img_ = Image.fromarray(cam).resize(self.input_size, Image.ANTIALIAS)
        cam = np.uint8(img_) / 255  # type: ignore

        return cam

    def apply_cmap_on_image(
        self, original_img: Image.Image, activation: np.ndarray, cmap_name: str = "rainbow"
    ):
        cmap = mpl_cm.get_cmap(cmap_name)
        no_trans_heatmap = cmap(activation)
        heatmap = copy.copy(no_trans_heatmap)
        heatmap[:, :, 3] = 0.4
        heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))  # type: ignore
        no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))  # type: ignore

        # convert to Image
        original_img = Image.fromarray(original_img)

        heatmap_on_image = Image.new("RGBA", original_img.size)
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, original_img.convert("RGBA"))
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
        return no_trans_heatmap, heatmap_on_image


def preprocess(
    path: T._path, input_size: Tuple[int, int] = (60, 60)
) -> Tuple[Tensor, np.ndarray]:
    raw_image = Image.open(str(path))
    raw_image = raw_image.resize(input_size)

    image = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )(raw_image)
    raw_image = np.array(raw_image)

    return image, raw_image  # type: ignore


@dataclass
class ExecuteGradCAM:
    classes: List[str]
    input_size: Tuple[int, int]
    target_layer: str
    device: torch.device
    schedule: List[bool]
    is_gradcam: bool = True

    @torch.enable_grad()
    def main(self, model: T._net, img_path: T._path) -> Dict[str, List[Image.Image]]:
        gcam = GradCAM(model, self.target_layer, self.input_size)
        processed_data = self.__get_init_dict()

        image, raw_image = preprocess(img_path, self.input_size)
        image = image.unsqueeze_(0).to(self.device)
        # raw_image = torch.from_numpy(RawIOBase).unsqueeze_(0).to(self.device)

        cam = gcam.generate_cam(image, self.device, target_cls=None)
        heatmap, heatmap_on_image = gcam.apply_cmap_on_image(raw_image, cam, "rainbow")

        processed_data["ggcam"].append(heatmap)
        processed_data["gcam"].append(heatmap_on_image)
        processed_data["gbp"].append(cam)
        return processed_data

    def __get_init_dict(self) -> Dict[str, list]:
        return {
            "gbp": [],  # Guided Back Propagation
            "gcam": [],  # Grad-CAM
            "ggcam": [],  # Guided Grad-CAM
        }
