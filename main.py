from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Tuple

import ignite.contrib.handlers.tensorboard_logger as tbl
from ignite.engine import Events
from ignite.engine.engine import Engine
from ignite.metrics import Accuracy, ConfusionMatrix, Loss
from ignite.metrics.running_average import RunningAverage
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

# my packages
import impl
import torch_utils as tutils
import utils
from gradcam import ExecuteGradCAM
from my_typings import State, T
from torch_utils import CreateDataset, get_dataloader
from yaml_parser import GlobalConfig, parse

gc: GlobalConfig


def run() -> None:
    if not Path(gc.path.dataset).exists():
        print(f"'{gc.path.dataset}' is not exist.")
        exit(-1)

    print(f"Creating dataset from '{gc.path.dataset}'...")
    dataset = CreateDataset(gc=gc)  # train, unknown, known
    train_loader, unknown_loader, known_loader = get_dataloader(
        dataset,
        input_size=gc.network.input_size,
        mini_batch=gc.network.batch,
        is_shuffle=gc.dataset.is_shuffle_per_epoch,
    )

    print("Building network...")
    classes = list(dataset.classes.values())
    device = gc.network.device

    net = gc.network.class_(input_size=gc.network.input_size, classify_size=len(classes)).to(device)
    model = impl.Model(
        net=net,
        optimizer=optim.Adam(net.parameters()),
        criterion=nn.CrossEntropyLoss(),
        device=device,
    )
    del net

    # logfile
    if gc.option.is_save_log:
        p = utils.create_filepath(gc.path.log, gc.filename_base, is_prefix_seq=True)
        gc.log = utils.LogFile(p, stdout=False)

    # netword difinition
    if gc.option.is_show_network_difinition:
        impl.show_network_difinition(gc, model, dataset)

    gcam = ExecuteGradCAM(
        classes,
        gc.network.input_size,
        gc.gradcam.layer,
        gpu_enabled=(gc.network.gpu_enabled and gc.gradcam.gpu_enabled),
        is_gradcam=gc.gradcam.enabled,
    )

    def train_step(engine: Engine, batch: T._batch_path_t):
        minibatch = tutils.MiniBatch(batch)
        return impl.train_step(minibatch, model, device, subdivisions=gc.network.subdivisions)

    def unknown_validation_step(engine: Engine, batch: T._batch_path_t):
        minibatch = tutils.MiniBatch(batch)
        epoch = engine.state.epoch
        return impl.validation_step(
            minibatch, model, device, epoch, gc, gcam, "unknown", non_blocking=False
        )

    def known_validation_step(engine: Engine, batch: T._batch_path_t):
        minibatch = tutils.MiniBatch(batch)
        epoch = engine.state.epoch
        return impl.validation_step(
            minibatch, model, device, epoch, gc, gcam, "known", non_blocking=False
        )

    # trainer / evaluator
    trainer = Engine(train_step)
    unknown_evaluator = Engine(unknown_validation_step)
    known_evaluator = Engine(known_validation_step)

    metrics = {
        "acc": Accuracy(),
        "loss": Loss(model.criterion),
        "cm": ConfusionMatrix(len(classes)),
    }
    utils.attach_metrics(unknown_evaluator, metrics)
    utils.attach_metrics(known_evaluator, metrics)

    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    # progress bar
    pbar = utils.MyProgressBar(persist=True, logfile=gc.log)
    pbar.attach(trainer, metric_names="all")

    collect_list = [
        (unknown_evaluator, unknown_loader, State.UNKNOWN_VALID),
        (known_evaluator, known_loader, State.KNOWN_VALID),
    ]

    tb_logger = None
    if gc.option.log_tensorboard:
        tb_logger = tbl.TensorboardLogger()

    # tensorboard
    if tb_logger and False:  # debug
        impl.log_tensorboard(tb_logger, trainer, collect_list, model)

    # schedule
    valid_schedule = utils.create_schedule(gc.network.epoch, gc.network.valid_cycle)
    save_schedule = utils.create_schedule(gc.network.epoch, gc.network.save_cycle)
    if not gc.network.is_save_final_model:
        save_schedule[-1] = False

    def validate_model(engine: Engine, collect_list: List[Tuple[Engine, DataLoader, str]]):
        if valid_schedule[engine.state.epoch - 1]:
            impl.validate_model(
                engine, collect_list, pbar, classes, tb_logger, verbose=gc.option.verbose,
            )

    def save_model(engine: Engine):
        epoch = engine.state.epoch
        if save_schedule[epoch - 1]:
            impl.save_model(model, classes, gc, epoch)

    # validate / save
    trainer.add_event_handler(Events.EPOCH_COMPLETED, validate_model, collect_list)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, save_model)

    # kick everything off
    trainer.run(train_loader, max_epochs=gc.network.epoch)

    if gc.option.log_tensorboard:
        tb_logger.close()


def parse_arg() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-c", "--cfg", help="config file", default="user_config.yaml")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arg()
    print(f"Loading config from '{args.cfg}'...")
    gc = parse(path=args.cfg)

    run()
