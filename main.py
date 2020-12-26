from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from typing import List, Tuple

import ignite.contrib.handlers.tensorboard_logger as tbl
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.nn as nn
import yaml
from ignite.engine import Events
from ignite.engine.engine import Engine
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import impl
from gradcam import ExecuteGradCAM
from modules import State, T
from modules import functions as fns
from modules import torch_utils as tutils
from modules import utils
from modules.global_config import GlobalConfig


def run() -> None:
    transform = Compose(
        [Resize(gc.network.input_size), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        # [Resize(gc.network.input_size), ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    cudnn.benchmark = True

    gc.check_dataset_path(is_show=True)
    dataset = tutils.CreateDataset(gc)  # train, unknown, known

    train_loader, unknown_loader, known_loader = dataset.get_dataloader(transform)

    if gc.option.is_save_config:
        dataset.write_config()  # write config of model
    del dataset.all_list

    classes = list(dataset.classes.values())
    device = gc.network.device

    print(f"Building network by '{gc.network.net_.__name__}'...")
    net = gc.network.net_(input_size=gc.network.input_size, classify_size=len(classes)).to(device)
    if isinstance(gc.network.optim_, optim.SGD):
        optimizer = gc.network.optim_(
            net.parameters(), lr=gc.network.lr, momentum=gc.network.momentum
        )
    else:
        optimizer = gc.network.optim_(net.parameters(), lr=gc.network.lr)

    model = tutils.Model(
        net,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        scaler=amp.GradScaler(enabled=gc.network.amp),
    )
    del net

    # logfile
    if gc.option.is_save_log:
        gc.logfile = utils.LogFile(gc.path.log.joinpath("log.txt"), stdout=False)
        gc.ratefile = utils.LogFile(gc.path.log.joinpath("rate.csv"), stdout=False)

        classes_ = ",".join(classes)
        gc.ratefile.writeline(f"epoch,known,{classes_},avg,,unknown,{classes_},avg")
        gc.ratefile.flush()

    # netword difinition
    impl.show_network_difinition(gc, model, dataset, stdout=gc.option.is_show_network_difinition)

    # grad cam
    gcam_schedule = (
        utils.create_schedule(gc.network.epoch, gc.gradcam.cycle)
        if gc.gradcam.enabled
        else [False] * gc.network.epoch
    )
    gcam = ExecuteGradCAM(
        classes,
        input_size=gc.network.input_size,
        target_layer=gc.gradcam.layer,
        device=device,
        schedule=gcam_schedule,
        is_gradcam=gc.gradcam.enabled,
    )

    # mkdir for gradcam
    phases = ["known", "unknown"]
    mkdir_options = {"parents": True, "exist_ok": True}
    for i, flag in enumerate(gcam_schedule):
        if not flag:
            continue
        ep_str = f"epoch{i+1}"
        for phase, cls in product(phases, classes):
            gc.path.gradcam.joinpath(f"{phase}_mistaken", ep_str, cls).mkdir(**mkdir_options)
        if not gc.gradcam.only_mistaken:
            for phase, cls in product(phases, classes):
                gc.path.gradcam.joinpath(f"{phase}_correct", ep_str, cls).mkdir(**mkdir_options)

    # progress bar
    pbar = utils.MyProgressBar(
        persist=True, logfile=gc.logfile, disable=gc.option.is_show_batch_result
    )

    # dummy functions
    exec_gcam_fn = fns.execute_gradcam if gc.gradcam.enabled else fns.dummy_execute_gradcam
    save_img_fn = (
        fns.save_mistaken_image
        if gc.option.is_save_mistaken_pred
        else fns.dummy_save_mistaken_image
    )
    exec_softmax_fn = (
        fns.execute_softmax if gc.option.is_save_softmax else fns.dummy_execute_softmax
    )

    def train_step(engine: Engine, batch: T._batch_path) -> float:
        return impl.train_step(
            engine,
            tutils.MiniBatch(batch),
            model,
            gc.network.subdivisions,
            gc,
            pbar,
            use_amp=gc.network.amp,
            save_img_fn=save_img_fn,
            non_blocking=True,
        )

    # trainer
    trainer = Engine(train_step)
    pbar.attach(trainer, metric_names="all")

    collect_list = [
        (known_loader, State.KNOWN),
        (unknown_loader, State.UNKNOWN),
    ]

    # tensorboard
    tb_logger = None
    if gc.option.log_tensorboard:
        tb_logger = tbl.TensorboardLogger(logdir=str(Path(gc.path.tb_log_dir, gc.filename_base)))
        impl.attach_log_to_tensorboard(tb_logger, trainer, collect_list, model)

    # schedule
    valid_schedule = utils.create_schedule(gc.network.epoch, gc.network.valid_cycle)
    save_schedule = utils.create_schedule(gc.network.epoch, gc.network.save_cycle)
    save_schedule[-1] = gc.network.is_save_final_model  # depends on config.

    save_cm_fn = fns.save_cm_image if gc.option.is_save_cm else fns.dummy_save_cm_image

    def validate_model(engine: Engine, collect_list: List[Tuple[DataLoader, str]]) -> None:
        epoch = engine.state.epoch
        # do not validate.
        if not (valid_schedule[epoch - 1] or gcam_schedule[epoch - 1]):
            return

        impl.validate_model(
            engine,
            collect_list,
            gc,
            pbar,
            classes,
            gcam=gcam,
            model=model,
            valid_schedule=valid_schedule,
            exec_gcam_fn=exec_gcam_fn,
            exec_softmax_fn=exec_softmax_fn,
            save_cm_fn=save_cm_fn,
            non_blocking=True,
        )

    def save_model(engine: Engine) -> None:
        epoch = engine.state.epoch
        if save_schedule[epoch - 1]:
            impl.save_model(model, classes, gc, epoch)

    # validate / save
    trainer.add_event_handler(Events.EPOCH_COMPLETED, validate_model, collect_list)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, save_model)

    # kick everything off
    trainer.run(train_loader, max_epochs=gc.network.epoch)

    # close file
    if gc.option.log_tensorboard:
        tb_logger.close()
    if gc.option.is_save_log:
        gc.logfile.close()
        gc.ratefile.close()


def parse_yaml(path: str) -> GlobalConfig:
    utils.check_existence(path)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        gc = GlobalConfig(data)
    return gc


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--cfg", help="config file", default="user_config.yaml")

    args = parser.parse_args()

    print(f"Loading config from '{args.cfg}'...")
    gc = parse_yaml(path=args.cfg)

    run()
