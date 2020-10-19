from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import ignite.contrib.handlers.tensorboard_logger as tbl
import yaml
from ignite.engine import Events
from ignite.engine.engine import Engine
from ignite.metrics import Accuracy, ConfusionMatrix, Loss
# from ignite.metrics.running_average import RunningAverage
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import impl
from gradcam import ExecuteGradCAM
from modules import State, T
from modules import torch_utils as tutils
from modules import utils
from modules.global_config import GlobalConfig


def run() -> None:
    transform = Compose(
        [Resize(gc.network.input_size), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    if gc.dataset.is_pre_splited:
        utils.check_existence([gc.dataset.train_dir, gc.dataset.valid_dir])
        print(f"Creating dataset from")
        print(f"  train - '{gc.dataset.train_dir}'")
        print(f"  valid - '{gc.dataset.valid_dir}'...")
    else:
        utils.check_existence(gc.path.dataset)
        print(f"Creating dataset from '{gc.path.dataset}'...")

    dataset = tutils.CreateDataset(gc)  # train, unknown, known
    if dataset.all_size == 0:
        raise ValueError(f"data size == 0, path of dataset is invalid.")

    train_loader, unknown_loader, known_loader = dataset.get_dataloader(transform)

    if gc.option.is_save_config:
        dataset.write_config()  # write config of model
    del dataset.all_list

    classes = list(dataset.classes.values())
    device = gc.network.device

    print(f"Building network by '{gc.network.net_.__name__}'...")
    net = gc.network.net_(input_size=gc.network.input_size, classify_size=len(classes)).to(device)
    model = impl.Model(
        net, optimizer=gc.network.optim_(net.parameters()), criterion=nn.CrossEntropyLoss(), device=device,
    )
    del net

    # logfile
    if gc.option.is_save_log:
        p = Path(gc.path.log, "log.txt")
        gc.logfile = utils.LogFile(p, stdout=False)

        p = Path(gc.path.log, "rate.csv")
        gc.ratefile = utils.LogFile(p, stdout=False)

        classes_ = ",".join(classes)
        gc.ratefile.writeline(f"epoch,known,{classes_},avg,,unknown,{classes_},avg")
        gc.ratefile.flush()

    # netword difinition
    impl.show_network_difinition(gc, model, dataset, stdout=gc.option.is_show_network_difinition)

    gcam = ExecuteGradCAM(
        classes,
        input_size=gc.network.input_size,
        target_layer=gc.gradcam.layer,
        device=device,
        is_gradcam=gc.gradcam.enabled,
    )

    def train_step(engine: Engine, batch: T._batch_path_t):
        minibatch = tutils.MiniBatch(batch)
        return impl.train_step(
            minibatch,
            model,
            subdivisions=gc.network.subdivisions,
            is_save_mistaken_pred=gc.option.is_save_mistaken_pred,
            non_blocking=True,
        )

    def unknown_validation_step(engine: Engine, batch: T._batch_path_t):
        minibatch = tutils.MiniBatch(batch)
        return impl.validation_step(
            engine, minibatch, model, gc, gcam, "unknown", non_blocking=True
        )

    def known_validation_step(engine: Engine, batch: T._batch_path_t):
        minibatch = tutils.MiniBatch(batch)
        return impl.validation_step(engine, minibatch, model, gc, gcam, "known", non_blocking=True)

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

    # progress bar
    pbar = utils.MyProgressBar(persist=True, logfile=gc.logfile)
    pbar.attach(trainer, metric_names="all")

    collect_list = [
        (known_evaluator, known_loader, State.KNOWN_VALID),
        (unknown_evaluator, unknown_loader, State.UNKNOWN_VALID),
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

    def validate_model(engine: Engine, collect_list: List[Tuple[Engine, DataLoader, str]]):
        if valid_schedule[engine.state.epoch - 1]:
            impl.validate_model(engine, collect_list, gc, pbar, classes, tb_logger)

    def save_model(engine: Engine):
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

    gc: GlobalConfig
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
