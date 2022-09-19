import argparse
from pathlib import Path
from datetime import datetime

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Average, Accuracy, RunningAverage
import torch
from torch.utils import tensorboard
import torch.nn.functional as F
import os
import numpy as np

from data import Dataset
from model import FcnResnet50


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using {} as device for training".format(device))

    kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {}

    # create log directory
    time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    description = "{}_dataset={},net={},lr={:.0e},{},threshold={},crop={}".format(
        time_stamp,
        args.dataset.name,
        args.net,
        args.lr,
        args.grasp_representation,
        args.metric_threshold,
        args.crop,
    ).strip(",")
    logdir = args.logdir / description

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.dataset, args.grasp_representation, args.metric_threshold,
        args.batch_size, args.val_split, args.crop, kwargs
    )

    # build the network
    centre_grasp_representation = args.grasp_representation == 'centre'
    net = FcnResnet50(centre_representation=centre_grasp_representation).to(device)

    # define optimizer and metrics
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    metrics = {
        "loss": Average(lambda out: out[3]),
        "accuracy": Accuracy(lambda out: (torch.round(out[1][0]), out[2][0])),
        "width_loss": Average(lambda out: out[4]),
        "rotation_loss": Average(lambda out: out[6]),
        "q_loss": Average(lambda out: out[5]),
        "centre_loss": Average(lambda out: out[7])
    }

    # create ignite engines for training and validation
    trainer = create_trainer(net, optimizer, loss_fn, metrics, device)
    evaluator = create_evaluator(net, loss_fn, metrics, device)

    # log training progress to the terminal and tensorboard
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'width_loss')
    RunningAverage(output_transform=lambda x: x[5]).attach(trainer, 'quality_loss')
    RunningAverage(output_transform=lambda x: x[6]).attach(trainer, 'rotation_loss')
    RunningAverage(output_transform=lambda x: x[7]).attach(trainer, 'centre_loss')

    if centre_grasp_representation:
        ProgressBar(persist=True, ascii=True).attach(trainer, ['loss', 'width_loss', 'quality_loss',
                                                               'rotation_loss', 'centre_loss'])
    else:
        ProgressBar(persist=True, ascii=True).attach(trainer, ['loss', 'width_loss', 'quality_loss',
                                                               'rotation_loss'])

    train_writer, val_writer = create_summary_writers(net, device, logdir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        epoch, metrics = trainer.state.epoch, trainer.state.metrics
        train_writer.add_scalar("loss", metrics["loss"], epoch)
        train_writer.add_scalar("quality_loss", metrics["q_loss"], epoch)
        train_writer.add_scalar("width_loss", metrics["width_loss"], epoch)
        train_writer.add_scalar("rotation_loss", metrics["rotation_loss"], epoch)
        train_writer.add_scalar("centre_loss", metrics["centre_loss"], epoch)
        train_writer.add_scalar("accuracy", metrics["accuracy"], epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, evaluator.state.metrics
        val_writer.add_scalar("loss", metrics["loss"], epoch)
        val_writer.add_scalar("quality_loss", metrics["q_loss"], epoch)
        val_writer.add_scalar("width_loss", metrics["width_loss"], epoch)
        val_writer.add_scalar("rotation_loss", metrics["rotation_loss"], epoch)
        val_writer.add_scalar("centre_loss", metrics["centre_loss"], epoch)
        val_writer.add_scalar("accuracy", metrics["accuracy"], epoch)

    # checkpoint model
    checkpoint_handler = ModelCheckpoint(
        logdir,
        n_saved=3,
        global_step_transform=global_step_from_engine(trainer),
        require_empty=True,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED(every=1), checkpoint_handler, {args.net: net}
    )

    # run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)

    torch.save(net.model.state_dict(), '{}/ros-gpnet.pt'.format(logdir), _use_new_zipfile_serialization=False)


def create_train_val_loaders(root, rep, threshold, batch_size, val_split, crop_image, kwargs):
    # load the dataset
    centre_rep = rep == 'centre'
    if crop_image:
        dataset = Dataset(root, centre_representation=centre_rep, metric_threshold=threshold, image_size=(300, 300))
    else:
        dataset = Dataset(root, centre_representation=centre_rep, metric_threshold=threshold)
    if 'val_indices.txt' in os.listdir(root):
        # We want to split according to the indices stored with our dataset
        val_indices = np.loadtxt('{}/val_indices.txt'.format(root), dtype=int)
        train_indices = np.loadtxt('{}/train_indices.txt'.format(root), dtype=int)
        train_set = torch.utils.data.Subset(dataset, train_indices)
        val_set = torch.utils.data.Subset(dataset, val_indices)
    else:
        print("Split indices not found in {}. Use random split.".format(root))
        # split into train and validation sets
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    return train_loader, val_loader

def prepare_batch(batch, device):
    depth_im, y_true = batch
    depth_im = depth_im.to(device)
    y_true = y_true.float().to(device)
    return depth_im, y_true

def select(pred, y_all):
    indices = torch.nonzero(y_all[:, :, :, 0])
    qual_pred = pred[0][indices[:, 0], 0, indices[:, 1], indices[:, 2]]
    rot_pred = pred[1][indices[:, 0], :, indices[:, 1], indices[:, 2]]
    params_pred = pred[2][indices[:, 0], :, indices[:, 1], indices[:, 2]]

    qual_gt = y_all[indices[:, 0], indices[:, 1], indices[:, 2], 1].float()
    rot_gt = y_all[indices[:, 0], indices[:, 1], indices[:, 2], 2:6].float()
    params_gt = y_all[indices[:, 0], indices[:, 1], indices[:, 2], 6:6+pred[2].shape[1]].float()
    return (qual_pred, rot_pred, params_pred), (qual_gt, rot_gt, params_gt)

def loss_fn(y_pred, y_true):
    label_pred, rotation_pred, params_pred = y_pred
    label, rotation, params = y_true
    loss_qual = _qual_loss_fn(label_pred, label)

    loss_rot = _quat_loss_fn(rotation_pred[torch.nonzero(label)][:, 0, :],
                             rotation[torch.nonzero(label)][:, 0, :])
    width_loss = _l1_loss_fn(params_pred[torch.nonzero(label), 0][:, 0], params[torch.nonzero(label), 0][:, 0])
    if params.shape[1] > 1:
        # TCP-based representation
        alpha = 0.01
        centre_loss = _l1_loss_fn(params_pred[torch.nonzero(label), 1][:, 0],
                                  params[torch.nonzero(label), 1][:, 0])
    else:
        # Contact-based representation
        alpha = 0.1
        centre_loss = torch.tensor(0.0)
    if loss_rot.shape[0] > 0:
        loss = loss_qual + alpha * width_loss + 0.1 * loss_rot.mean() + 0.1 * centre_loss
        return loss, width_loss, loss_qual, loss_rot.mean(), centre_loss
    else:
        # No ground-truth positive grasp contact visible - ignore orientation & width loss, since
        # they would be None.
        loss = loss_qual
        return loss, torch.tensor(0.0), loss_qual, torch.tensor(0.0), torch.tensor(0.0)


def _qual_loss_fn(pred, target):
    return F.binary_cross_entropy(pred, target, reduction="mean")

def _quat_loss_fn(pred, target):
    return 1.0 - torch.abs(torch.sum(pred * target, dim=1))

def _l1_loss_fn(pred, target):
    return F.l1_loss(pred, target, reduction="mean")

def create_trainer(net, optimizer, loss_fn, metrics, device):
    def _update(_, batch):
        net.train()
        optimizer.zero_grad()

        # forward
        x, y = prepare_batch(batch, device)
        y_pred, y_true = select(net(x), y)
        loss, width_loss, q_loss, rot_loss, centre_loss = loss_fn(y_pred, y_true)

        # backward
        loss.backward()
        optimizer.step()

        return x, y_pred, y_true, loss, width_loss, q_loss, rot_loss, centre_loss

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer

def create_evaluator(net, loss_fn, metrics, device):
    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device)
            y_pred, y_true = select(net(x), y)
            loss, width_loss, q_loss, rot_loss, centre_loss = loss_fn(y_pred, y_true)

        return x, y_pred, y_true, loss, width_loss, q_loss, rot_loss, centre_loss

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator

def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    return train_writer, val_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="conv")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--grasp-representation", type=str, choices=["contact", "centre"], default="contact")
    parser.add_argument("--metric-threshold", type=float, default=0.5)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--crop", action='store_true')
    parser.add_argument("--no-crop", dest='crop', action='store_false')
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.set_defaults(crop=True)
    args = parser.parse_args()
    main(args)
