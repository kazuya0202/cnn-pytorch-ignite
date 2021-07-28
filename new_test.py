from __future__ import annotations
from gradcam.gcam import GradCamType

from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.cuda
import yaml
from ignite.metrics.accuracy import Accuracy
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.metrics.metric import Metric
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from texttable import Texttable
import cnn
from gradcam import ExecuteGradCAM
from modules import torch_utils as tutils
from modules import utils
from test_config import Config


def make_dataset(root: Path, phase: str) -> list[tutils.Data]:
    dirs = sorted(list(root.glob("*")))
    exts = ["jpg", "png", "jpeg", "gif", "bmp"]

    data_list = []
    for label_idx, child_dir in enumerate(dirs):
        tmp_list = [
            tutils.Data(str(x), label_idx, child_dir.name)
            for ext in exts
            for x in child_dir.glob(f"*.{ext}")
            if x.is_file()
        ]
        data_list.extend(tmp_list)

    print(f"Totally {len(data_list)} samples in {phase} set.")
    return data_list


class CustomDataset(Dataset):
    def __init__(self, root: Path, transform: Optional[Callable], phase: str = "test") -> None:
        self.transform = transform
        self.data_list = make_dataset(root, phase)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index) -> tuple[Any, int, str]:
        data = self.data_list[index]
        img_path, label, _ = data.items()
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, label, img_path


if __name__ == "__main__":
    cudnn.benchmark = True

    # * load config
    cfg = Config()
    cfg.update(yaml.safe_load(open("./user_config_for_test.yaml", encoding="utf-8")))

    ckpt = torch.load(cfg.model_path)
    classes: list[str] = ckpt["classes"]
    input_size = (cfg.width, cfg.height)

    phase = "test"
    model_name = Path(cfg.model_path).name
    model_stem = Path(cfg.model_path).stem

    # * prepare network
    net = cnn.Net(
        input_size=input_size,
        classify_size=len(classes),
        in_channels=cfg.channels,
    )
    net.load_state_dict(ckpt["model_state_dict"])

    use_gpu = torch.cuda.is_available() and cfg.gpu_enabled
    device = torch.device("cuda" if use_gpu else "cpu")

    net.to(device)
    net.eval()

    # * prepare grad-cam
    gcam = ExecuteGradCAM(
        classes,
        input_size=input_size,
        target_layer=cfg.gradcam.layer,
        device=device,
        schedule=[True],  # only once
        is_gradcam=cfg.gradcam.enabled,
    )

    mkdir_options = dict(parents=True, exist_ok=True)

    save_root = Path(cfg.save_root)
    gcam_save_root = save_root / model_stem

    # make direcotories
    for class_name in classes:
        (gcam_save_root / "mistaken" / class_name).mkdir(**mkdir_options)
        (gcam_save_root / "correct" / class_name).mkdir(**mkdir_options)

    # * prepare dataset, dataloader
    transform = Compose(
        [
            Resize((cfg.width, cfg.height)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = CustomDataset(Path(cfg.img_root), transform, phase=phase)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # table
    table = Texttable()
    table.header(["No.", "Name", "Accuracy", "Detail"])
    table.set_deco(Texttable.HEADER)
    table.set_cols_align(["c", "l", "r", "l"])

    # softmax
    softmax_file = open(save_root / "softmax.csv", "w")

    classes_sep = ",".join([f"[{cls}]" for cls in classes])
    softmax_file.write(f"correct class,predict class,filename,,softmax,{classes_sep}\n")

    metrics: dict[str, Metric] = {}

    # compute only validation.
    metrics = {
        "acc": Accuracy(device=device),
        "cm": ConfusionMatrix(len(classes), device=device),
    }

    for x, y, path in tqdm(loader):
        with torch.no_grad():
            x, y = utils.prepare_batch((x, y), device, non_blocking=True)
            # x = x.unsqueeze_(0)
            y_pred = net(x)
        ans, pred = utils.get_label(y, y_pred)

        # path = str(path[0])
        path = Path(path[0])
        stem = path.stem

        # softmax processing
        pred_softmax = y_pred.softmax(dim=1)[0]
        softmax_list = list(map(lambda x: str(round(x.item(), 5)), pred_softmax))
        correct_cls = classes[int(y.item())]
        pred_cls = classes[int(torch.argmax(pred_softmax).item())]
        onehot = ",".join(softmax_list)
        softmax_file.write(f"{correct_cls},{pred_cls},{stem},,,{onehot}\n")

        # gradcam processing
        is_correct = ans == pred
        dir_name = "correct" if is_correct else "mistaken"
        class_name = classes[ans]

        ret = gcam.process_single_image(net, path)
        ret.pop(GradCamType.CAM)  # ignore cam

        for phase, data in ret.items():
            filename = f"{stem}_pred[{pred}]_ans[{ans}]_{phase}.jpg"
            output_fp = gcam_save_root / dir_name / class_name / filename
            img = data.convert("RGB")
            img.save(output_fp)

        # update metrics
        for metric in metrics.values():
            metric.update(output=(y_pred, y))

    # calculate metrics
    avg_acc = metrics["acc"].compute()
    cm = utils.tensor2np(metrics["cm"].compute())

    # confusion matrix
    title = f"Confusion Matrix ({model_name})"
    fig = utils.plot_confusion_matrix(cm, classes, title=title)
    output_path = save_root / f"{model_stem}_cm.jpg"
    fig.savefig(str(output_path))

    for i, cls_name in enumerate(classes):
        n_all = sum(cm[i])
        n_acc = cm[i][i]
        acc = n_acc / n_all

        table.add_row([i, cls_name, acc, f"{n_acc} / {n_all} images"])
    table.add_row(["*", "[avg]", round(avg_acc, 3), "-"])
    print(f"\n{table.draw()}\n")
