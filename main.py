from typing import Tuple
import ignite.contrib.handlers.tensorboard_logger as tbl
from ignite.engine.engine import Engine
from ignite.metrics.running_average import RunningAverage

import torch
from ignite.engine import Events

# from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, ConfusionMatrix
from torch import Tensor, nn
from torch.optim import Adam

# my packages
import cnn
from torch_utils import CreateDataset, get_data_loader
from yaml_parser import GlobalConfig, parse
import utils
import impl
from my_typings import Names, T

gc: GlobalConfig


def run():
    print("Creating dataset...")
    dataset = CreateDataset(GCONF=gc)  # train, unknown, known
    train_loader, unknown_loader, known_loader = get_data_loader(
        dataset,
        input_size=gc.network.input_size,
        mini_batch=gc.network.batch,
        is_shuffle=gc.network.is_shuffle_dataset_per_epoch,
    )

    print("Building network...")
    model = cnn.Net(input_size=gc.network.input_size, classify_size=len(dataset.classes))
    device = torch.device("cuda")

    model.to(device=device)
    optimizer = Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    def train_step(engine: Engine, batch: Tuple[Tensor, Tensor]):
        return impl.train_step(
            batch, model, optimizer, criterion, device, subdivision=gc.network.subdivision
        )

    def validation_step(engine: Engine, batch: T._BATCH):
        return impl.validation_step(batch, model, device, non_blocking=False)

    # trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    trainer = Engine(train_step)

    metrics = {"acc": Accuracy(), "loss": Loss(criterion)}
    metrics_with_cm = {
        "acc": Accuracy(),
        "loss": Loss(criterion),
        "cm": ConfusionMatrix(len(dataset.classes)),
    }

    # train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    # unknown_evaluator = create_supervised_evaluator(model, metrics=metrics_with_cm, device=device)
    # known_evaluator = create_supervised_evaluator(model, metrics=metrics_with_cm, device=device)
    train_evaluator = Engine(validation_step)
    unknown_evaluator = Engine(validation_step)
    known_evaluator = Engine(validation_step)

    utils.attach_metrics(train_evaluator, metrics)
    utils.attach_metrics(unknown_evaluator, metrics_with_cm)
    utils.attach_metrics(known_evaluator, metrics_with_cm)

    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    pbar = utils.MyProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    tb_logger = tbl.TensorboardLogger()
    if False:
        attach_num = 1
        tb_logger = tbl.TensorboardLogger()
        # tb_logger = tbl.TensorboardLogger(log_dir=gc.path.tensorboard_log)

        tb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED(every=attach_num),
            tag="training",  # type: ignore
            output_transform=lambda loss: {"batchloss": loss},  # type: ignore
            metric_names="all",  # type: ignore
        )

        _list = [
            ("training", train_evaluator),
            ("unknown validation", unknown_evaluator),
            ("known validation", known_evaluator),
        ]
        for tag, evaluator in _list:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=["loss", "acc"],  # type: ignore
                global_step_transform=tbl.global_step_from_engine(trainer),  # type: ignore
            )

        tb_logger.attach_opt_params_handler(
            trainer, event_name=Events.ITERATION_COMPLETED(every=attach_num), optimizer=optimizer  # type: ignore
        )

        tb_logger.attach(
            trainer,
            log_handler=tbl.WeightsScalarHandler(model),
            event_name=Events.ITERATION_COMPLETED(every=attach_num),
        )

        tb_logger.attach(
            trainer,
            log_handler=tbl.WeightsHistHandler(model),
            event_name=Events.EPOCH_COMPLETED(every=attach_num),
        )

        tb_logger.attach(
            trainer,
            log_handler=tbl.GradsScalarHandler(model),
            event_name=Events.ITERATION_COMPLETED(every=attach_num),
        )

        tb_logger.attach(
            trainer,
            log_handler=tbl.GradsHistHandler(model),
            event_name=Events.EPOCH_COMPLETED(every=attach_num),
        )

    # def score_function(engine):
    #     return engine.state.metrics["accuracy"]

    # model_checkpoint = ModelCheckpoint(
    #     log_dir,
    #     n_saved=2,
    #     filename_prefix="best",
    #     score_function=score_function,
    #     score_name="validation_accuracy",
    #     global_step_transform=global_step_from_engine(trainer),
    # )
    # validation_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine: Engine):
        _list = [
            (train_evaluator, train_loader, Names.TRAINING),
            (unknown_evaluator, unknown_loader, Names.UNKNOWN_VALID),
            (known_evaluator, known_loader, Names.KNOWN_VALID),
        ]
        impl.log_results(
            engine,
            _list,
            pbar,
            classes=list(dataset.classes.values()),
            tb_logger=tb_logger,
            verbose=gc.option.verbose,
        )

    # kick everything off
    trainer.run(train_loader, max_epochs=3)

    tb_logger.close()


if __name__ == "__main__":
    path = "./user_config.yaml"
    gc = parse(path=path)

    run()
