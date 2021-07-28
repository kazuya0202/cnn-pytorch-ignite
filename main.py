from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from typing import List, Tuple

import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.nn as nn
import yaml
from ignite.engine import Events
from ignite.engine.engine import Engine
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
        [Resize(cfg.network.input_size), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        # [Resize(gc.network.input_size), ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    cudnn.benchmark = True

    cfg.check_dataset_path(is_show=True)
    dataset = tutils.CreateDataset(cfg)  # train, unknown, known

    train_loader, unknown_loader, known_loader = dataset.get_dataloader(transform)

    if cfg.option.is_save_config:
        dataset.write_config()  # write config of model
    del dataset.all_list

    classes = list(dataset.classes.values())
    device = cfg.network.device

    print(f"Building network by '{cfg.network.net_.__name__}'...")
    net = cfg.network.net_(input_size=cfg.network.input_size, classify_size=len(classes)).to(device)
    if isinstance(cfg.network.optim_, optim.SGD):
        optimizer = cfg.network.optim_(
            net.parameters(), lr=cfg.network.lr, momentum=cfg.network.momentum
        )
    else:
        optimizer = cfg.network.optim_(net.parameters(), lr=cfg.network.lr)

    model = tutils.Model(
        net,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        scaler=amp.GradScaler(enabled=cfg.network.amp),
    )
    del net, optimizer

    # logfile
    if cfg.option.is_save_log:
        cfg.logfile = utils.LogFile(cfg.path.log.joinpath("log.txt"), stdout=False)
        cfg.ratefile = utils.LogFile(cfg.path.log.joinpath("rate.csv"), stdout=False)

        classes_ = ",".join(classes)
        cfg.ratefile.writeline(f"epoch,known,{classes_},avg,,unknown,{classes_},avg")
        cfg.ratefile.flush()

    # netword difinition
    impl.show_network_difinition(cfg, model, dataset, stdout=cfg.option.is_show_network_difinition)

    # grad cam
    gcam_schedule = (
        utils.create_schedule(cfg.network.epoch, cfg.gradcam.cycle)
        if cfg.gradcam.enabled
        else [False] * cfg.network.epoch
    )
    gcam = ExecuteGradCAM(
        classes,
        input_size=cfg.network.input_size,
        target_layer=cfg.gradcam.layer,
        device=device,
        schedule=gcam_schedule,
        is_gradcam=cfg.gradcam.enabled,
    )

    # mkdir for gradcam
    phases = ["known", "unknown"]
    mkdir_options = {"parents": True, "exist_ok": True}
    for i, flag in enumerate(gcam_schedule):
        if not flag:
            continue
        ep_str = f"epoch{i+1}"
        for phase, cls in product(phases, classes):
            cfg.path.gradcam.joinpath(f"{phase}_mistaken", ep_str, cls).mkdir(**mkdir_options)
        if not cfg.gradcam.only_mistaken:
            for phase, cls in product(phases, classes):
                cfg.path.gradcam.joinpath(f"{phase}_correct", ep_str, cls).mkdir(**mkdir_options)

    # progress bar
    pbar = utils.MyProgressBar(
        persist=True, logfile=cfg.logfile, disable=cfg.option.is_show_batch_result
    )

    # dummy functions
    exec_gcam_fn = fns.execute_gradcam if cfg.gradcam.enabled else fns.dummy_execute_gradcam
    save_img_fn = (
        fns.save_mistaken_image
        if cfg.option.is_save_mistaken_pred
        else fns.dummy_save_mistaken_image
    )
    exec_softmax_fn = (
        fns.execute_softmax if cfg.option.is_save_softmax else fns.dummy_execute_softmax
    )

    def train_step(engine: Engine, batch: T._batch_path) -> float:
        return impl.train_step(
            engine,
            tutils.MiniBatch(batch),
            model,
            cfg.network.subdivisions,
            cfg,
            pbar,
            use_amp=cfg.network.amp,
            save_img_fn=save_img_fn,
            non_blocking=True,
        )

    # trainer
    trainer = Engine(train_step)
    pbar.attach(trainer, metric_names="all")

    # tensorboard logger
    tb_logger = (
        SummaryWriter(log_dir=str(Path(cfg.path.tb_log_dir, cfg.filename_base)))
        if cfg.option.log_tensorboard
        else None
    )

    # schedule
    valid_schedule = utils.create_schedule(cfg.network.epoch, cfg.network.valid_cycle)
    save_schedule = utils.create_schedule(cfg.network.epoch, cfg.network.save_cycle)
    save_schedule[-1] = cfg.network.is_save_final_model  # depends on config.

    save_cm_fn = fns.save_cm_image if cfg.option.is_save_cm else fns.dummy_save_cm_image

    def validate_model(engine: Engine, collect_list: List[Tuple[DataLoader, str]]) -> None:
        epoch = engine.state.epoch
        # do not validate.
        if not (valid_schedule[epoch - 1] or gcam_schedule[epoch - 1]):
            return

        impl.validate_model(
            engine,
            collect_list,
            cfg,
            pbar,
            classes,
            gcam=gcam,
            model=model,
            valid_schedule=valid_schedule,
            tb_logger=tb_logger,
            exec_gcam_fn=exec_gcam_fn,
            exec_softmax_fn=exec_softmax_fn,
            save_cm_fn=save_cm_fn,
            non_blocking=True,
        )

    def save_model(engine: Engine) -> None:
        epoch = engine.state.epoch
        if save_schedule[epoch - 1]:
            impl.save_model(model, classes, cfg, epoch)

    # validate / save
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        validate_model,
        [(known_loader, State.KNOWN), (unknown_loader, State.UNKNOWN)],  # args
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, save_model)

    # kick everything off
    trainer.run(train_loader, max_epochs=cfg.network.epoch)

    # close file
    if cfg.option.log_tensorboard and tb_logger:
        tb_logger.close()
    if cfg.option.is_save_log:
        cfg.logfile.close()
        cfg.ratefile.close()


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
    cfg = parse_yaml(path=args.cfg)

    run()
