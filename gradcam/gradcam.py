from dataclasses import dataclass
from typing import List, Tuple, Union

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
from modules.my_typings import T
from PIL import Image
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm

from . import BackPropagation, Deconvnet, GradCAM, GuidedBackPropagation


def preprocess(image_path: str, input_size: tuple) -> Tuple[Tensor, Tensor]:
    r"""Load image, convert its type to torch.Tensor and return it.

    Args:
        image_path (str): path of image.
        input_size (tuple): input image size.

    Returns:
        Tuple: image data.
    """

    # cv2 can't load Japanese filename.
    # raw_image = cv2.imread(image_path)  # type: ignore

    # load using pillow instead of cv2.
    img_pil = np.array(Image.open(image_path))
    raw_image = cv2.cvtColor(img_pil, cv2.COLOR_RGB2BGR)  # type: ignore
    raw_image = cv2.resize(raw_image, input_size)  # type: ignore

    image = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )(raw_image[..., ::-1].copy())

    return image, raw_image


def load_images(
    image_paths: List[str], input_size: tuple = (60, 60)
) -> Tuple[List[Tensor], List[Tensor]]:
    r"""Load images.

    Returns:
        Tuple: image data.
    """

    images = []
    raw_images = []

    for image_path in image_paths:
        image, raw_image = preprocess(image_path, input_size)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def get_data_of_gradient(gradient: Tensor) -> Tensor:
    r"""Returns gradient data.

    Args:
        gradient (Tensor): gradient.

    Returns:
        Tensor: calculated gradient data.
    """
    np_gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    del gradient
    np_gradient -= np.min(np_gradient)
    np_gradient /= np.max(np_gradient)
    np_gradient *= 255.0

    np_gradient = np.uint8(np_gradient)  # type: ignore
    return np_gradient


def get_data_of_gradcam(gcam: Tensor, raw_image: Tensor, paper_cmap: bool = False) -> Tensor:
    r"""Returns Grad-CAM data.

    Args:
        gcam (Tensor): Grad-CAM data.
        raw_image (Tensor): raw image data.
        paper_cmap (bool, optional): cmap. Defaults to False.

    Returns:
        Tensor: [description]
    """
    np_gcam = gcam.cpu().numpy()
    del gcam
    cmap = cm.jet_r(np_gcam)[..., :3] * 255.0  # type: ignore
    if paper_cmap:
        alpha = np_gcam[..., None]
        np_gcam = alpha * cmap + ([1] - alpha) * raw_image
        # np_gcam = alpha * cmap + ([torch.tensor(1)] - alpha) * raw_image
    else:
        np_gcam = (cmap.astype(np.float) + raw_image.clone().cpu().numpy().astype(np.float)) / 2

    np_gcam = np.uint8(np_gcam)  # type: ignore
    return np_gcam


@dataclass
class ExecuteGradCAM:
    def __init__(
        self,
        classes: List[str],
        input_size: Tuple[int, int],
        target_layer: str,
        gpu_enabled: bool = True,
        **options
    ) -> None:
        r"""
        Args:
            classes (List[str]): classes of model.
            input_size (Tuple[int, int]): input image size.
            target_layer (str): grad cam layer.

        **options
            is_vanilla (bool): execute `Vanilla`. Defaults to False.
            is_deconv (bool): execute `Deconv Net`. Defaults to False.
            is_gradcam (bool): execute `Grad-CAM`. Defaults to False.
        """

        self.classes = classes
        self.input_size = input_size
        self.target_layer = target_layer
        self.gpu_enabled = gpu_enabled

        self.device = None

        self.is_vanilla = options.pop("is_vanilla", False)
        self.is_deconv = options.pop("is_deconv", False)
        self.is_gradcam = options.pop("is_gradcam", False)

        self.class_num = len(classes)

    @torch.enable_grad()  # enable gradient
    def main(self, model: T._net_t, image_path: Union[tuple, list, str]) -> dict:
        """Switch execute function.

        Args:
            model (T._net_t): model.
            image_path (Union[list, str]): path of image.

        Returns:
            List: processed image data.
        """
        model.eval()  # switch to eval

        restore_device = None
        if not self.gpu_enabled:
            restore_device = next(model.parameters()).device
            model.to(torch.device("cpu"))  # use only cpu

        self.device = next(model.parameters()).device
        ret = {}

        # convert to list
        if isinstance(image_path, tuple):
            image_path = list(image_path)

        # process one image.
        if isinstance(image_path, str):
            ret = self._execute_one_image(model, image_path)

        # process multi images.
        elif isinstance(image_path, list):
            ret = self._execute_multi_images(model, image_path)

        if not self.gpu_enabled:
            model.to(restore_device)
        return ret.copy()

    def _execute_one_image(self, model: T._net_t, image_path: str) -> dict:
        """Process one image.

        Args:
            model (T._net_t): model.
            image_path (str): path of image.
        """
        processed_data = {
            "vanilla": [],  # Vanilla
            "deconv": [],  # Deconv Net
            "gbp": [],  # Guided Back Propagation
            "gcam": [],  # Grad-CAM
            "ggcam": [],  # Guided Grad-CAM
        }

        # device = next(model.parameters()).device  # get device
        if not self.gpu_enabled:
            self.device = torch.device("cpu")  # cpu only

        image, raw_image = preprocess(image_path, self.input_size)
        image = image.unsqueeze_(0).to(self.device)
        raw_image = torch.from_numpy(raw_image)
        raw_image = raw_image.unsqueeze_(0).to(self.device)

        # --- Vanilla Backpropagation ---
        bp = BackPropagation(model=model)
        _, ids = bp.forward(image)  # sorted

        # --- Deconvolution ---
        deconv = None

        if self.is_deconv:
            deconv = Deconvnet(model=model)
            _ = deconv.forward(image)

        # --- Grad-CAM / Guided Backpropagation / Guided Grad-CAM ---
        gcam = None
        gbp = None

        if self.is_gradcam:
            gcam = GradCAM(model=model)
            _ = gcam.forward(image)
            del _

            gbp = GuidedBackPropagation(model=model)
            _ = gbp.forward(image)
            del _
        return processed_data

        pbar = tqdm(
            range(self.class_num),
            total=self.class_num,
            ncols=100,
            bar_format="{l_bar}{bar:30}{r_bar}",
            leave=False,
        )
        pbar.set_description("Grad-CAM")

        for i in pbar:
            if self.is_vanilla:
                bp.backward(ids=ids[:, [i]])
                gradients = bp.generate()

                # append
                data = get_data_of_gradient(gradients)
                processed_data["vanilla"].append(data)
                del gradients, data

            if self.is_deconv:
                deconv.backward(ids=ids[:, [i]])
                gradients = deconv.generate()

                # append
                data = get_data_of_gradient(gradients)
                processed_data["deconv"].append(data)
                del gradients, data

            # Grad-CAM / Guided Grad-CAM / Guided Backpropagation
            if self.is_gradcam:
                gbp.backward(ids=ids[:, [i]])
                gradients = gbp.generate()

                # Grad-CAM
                gcam.backward(ids=ids[:, [i]])
                regions = gcam.generate(target_layer=self.target_layer)

                # append
                data = get_data_of_gradient(gradients[0])
                processed_data["gbp"].append(data)

                data = get_data_of_gradcam(regions[0, 0], raw_image[0])
                processed_data["gcam"].append(data)

                data = get_data_of_gradient(torch.mul(regions, gradients)[0])
                processed_data["ggcam"].append(data)
                del gradients, regions, data

        # Remove all the hook function in the 'model'
        bp.remove_hook()

        if self.is_deconv:
            deconv.remove_hook()

        if self.is_gradcam:
            gcam.remove_hook()
            gbp.remove_hook()

        del ids, bp, gbp, gcam, image, raw_image
        return processed_data.copy()

    def _execute_multi_images(self, model: T._net_t, image_paths: List[str]) -> dict:
        r"""Process multiple images.

        Args:
            model (T._net_t): model.
            image_paths (List[str]): path of images.
        """
        processed_data = {
            "vanilla": [],  # Vanilla
            "deconv": [],  # Deconv Net
            "gbp": [],  # Guided Back Propagation
            "gcam": [],  # Grad-CAM
            "ggcam": [],  # Guided Grad-CAM
        }

        # device = next(model.parameters()).device  # get device
        if not self.gpu_enabled:
            self.device = torch.device("cpu")  # only cpu

        images, raw_images = load_images(image_paths, self.input_size)
        images = torch.stack(images).to(self.device)

        # --- Vanilla Backpropagation ---
        bp = BackPropagation(model=model)
        _, ids = bp.forward(images)  # sorted

        # --- Deconvolution ---
        deconv = None

        if self.is_deconv:
            deconv = Deconvnet(model=model)
            _ = deconv.forward(images)

        # --- Grad-CAM / Guided Backpropagation / Guided Grad-CAM ---
        gcam = None
        gbp = None

        if self.is_gradcam:
            gcam = GradCAM(model=model)
            _ = gcam.forward(images)

            gbp = GuidedBackPropagation(model=model)
            _ = gbp.forward(images)

        pbar = tqdm(
            range(self.class_num),
            total=self.class_num,
            ncols=100,
            bar_format="{l_bar}{bar:30}{r_bar}",
            leave=False,
        )
        pbar.set_description("Grad-CAM")

        for i in pbar:
            if self.is_vanilla:
                bp.backward(ids=ids[:, [i]])
                gradients = bp.generate()

                # Save results as image files
                for j in range(len(images)):
                    # append
                    data = get_data_of_gradient(gradients[j])
                    processed_data["vanilla"].append(data)

            if self.is_deconv:
                deconv.backward(ids=ids[:, [i]])
                gradients = deconv.generate()

                for j in range(len(images)):
                    # append
                    data = get_data_of_gradient(gradients[j])
                    processed_data["deconv"].append(data)

            # Grad-CAM / Guided Grad-CAM / Guided Backpropagation
            if self.is_gradcam:
                gbp.backward(ids=ids[:, [i]])
                gradients = gbp.generate()

                # Grad-CAM
                gcam.backward(ids=ids[:, [i]])
                regions = gcam.generate(target_layer=self.target_layer)

                for j in range(len(images)):
                    # append
                    data = get_data_of_gradient(gradients[j])
                    processed_data["gbp"].append(data)

                    data = get_data_of_gradcam(regions[j, 0], raw_images[j])
                    processed_data["gcam"].append(data)

                    data = get_data_of_gradient(torch.mul(regions, gradients)[j])
                    processed_data["ggcam"].append(data)

        # Remove all the hook function in the 'model'
        bp.remove_hook()

        if self.is_deconv:
            deconv.remove_hook()

        if self.is_gradcam:
            gcam.remove_hook()
            gbp.remove_hook()

        return processed_data


@dataclass
class ExecuteOnlyGradCAM:
    classes: List[str]
    input_size: Tuple[int, int]
    target_layer: str
    gpu_enabled: bool = True
    is_gradcam: bool = True

    def __post_init__(self) -> None:
        self.device = None
        self.class_num = len(self.classes)

    def __get_init_dict(self) -> dict:
        return {
            "gbp": [],  # Guided Back Propagation
            "gcam": [],  # Grad-CAM
            "ggcam": [],  # Guided Grad-CAM
        }

    @torch.enable_grad()  # enable gradient
    def main(self, model: T._net_t, image_path: T._path_t) -> dict:
        """Switch execute function.

        Args:
            model (T._net_t): model.
            image_path (Union[list, str]): path of image.

        Returns:
            List: processed image data.
        """
        if not self.is_gradcam:
            return {}

        model.eval()  # switch to eval
        model.zero_grad()

        restore_device = None
        if not self.gpu_enabled:
            restore_device = next(model.parameters()).device
            model.to(torch.device("cpu"))  # use only cpu

        self.device = next(model.parameters()).device

        ret = self._execute_one_image(model, str(image_path))

        if not self.gpu_enabled:
            model.to(restore_device)
        return ret.copy()

    def _execute_one_image(self, model: T._net_t, image_path: str) -> dict:
        """Process one image.

        Args:
            model (T._net_t): model.
            image_path (str): path of image.
        """
        processed_data = self.__get_init_dict()

        # device = next(model.parameters()).device  # get device
        if not self.gpu_enabled:
            self.device = torch.device("cpu")  # cpu only

        image, raw_image = preprocess(image_path, self.input_size)
        image = image.unsqueeze_(0).to(self.device)
        raw_image = torch.from_numpy(raw_image)
        raw_image = raw_image.unsqueeze_(0).to(self.device)

        # --- Vanilla Backpropagation ---
        bp = BackPropagation(model=model)
        _, ids = bp.forward(image)  # sorted

        # --- Grad-CAM / Guided Backpropagation / Guided Grad-CAM ---
        gcam = GradCAM(model=model)
        gcam.forward(image)

        gbp = GuidedBackPropagation(model=model)
        gbp.forward(image)

        pbar = tqdm(
            range(self.class_num),
            total=self.class_num,
            # ncols=100,
            # bar_format="{l_bar}{bar:30}{r_bar}",
            leave=False,
        )
        pbar.set_description("Grad-CAM")

        for i in pbar:
            # Grad-CAM / Guided Grad-CAM / Guided Backpropagation
            gbp.backward(ids=ids[:, [i]])
            gradients = gbp.generate().clone().cpu().numpy()

            # Grad-CAM
            gcam.backward(ids=ids[:, [i]])
            regions = gcam.generate(target_layer=self.target_layer).clone().cpu().numpy()

            # append
            data = get_data_of_gradient(gradients[0]).clone().cpu().numpy()
            processed_data["gbp"].append(data)

            data = get_data_of_gradcam(regions[0, 0], raw_image[0]).clone().cpu().numpy()
            processed_data["gcam"].append(data)

            data = get_data_of_gradient(torch.mul(regions, gradients)[0]).clone().cpu().numpy()
            processed_data["ggcam"].append(data)
            del gradients, regions, data

        # Remove all the hook function in the 'model'
        bp.remove_hook()
        gcam.remove_hook()
        gbp.remove_hook()

        # TODO
        model.zero_grad()
        for x in model.modules():
            x.zero_grad()
        del ids, bp, gbp, gcam, image, raw_image
        return processed_data.copy()
