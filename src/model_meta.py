import torchmetrics
import torch
from torch.utils.tensorboard import SummaryWriter


class ModelMetrics:
    def __init__(
        self,
        device: torch.device,
        single_form="{}: {:.2f} | ",
        compute_form="  -: {}: {:.2f}\n",
        batch_epoch_form="[B{:02d}  E{:02d}] | ",
        comment="",
    ):
        self.metrics = {}
        self.device = device
        self.single_form = single_form
        self.compute_form = compute_form
        self.batch_epoch_form = batch_epoch_form

        self.current_step = None
        self.current_epoch = None

        self.loss = torchmetrics.MeanMetric()

        if comment == "":
            self.writer = SummaryWriter()
        else:
            comment = comment.replace(" ", "_")
            self.writer = SummaryWriter(comment=comment)

    def format_single(self, key, value, form=None):
        if form is None:
            form = self.single_form
        return form.format(key, value)

    def format_compute(self, key, value, form=None):
        if form is None:
            form = self.compute_form
        return form.format(key, value)

    def add_metric(self, name, metric):
        self.metrics[name] = metric.to(self.device)

    # runtime methods

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

        self.current_step = None
        self.current_epoch = None

    def update(self, y_pred, y_true, loss, show=False, step=None, epoch=None):
        self.current_step = step
        self.current_epoch = epoch

        self.loss.update(loss)
        self.writer.add_scalar("loss", loss, epoch)

        result = {}
        for name, metric in self.metrics.items():
            result[name] = metric(y_pred, y_true)
            if epoch != None:
                self.writer.add_scalar(name, result[name], epoch)
        if show:
            self.show(result)
        return result

    def compute(self, show=False):
        result = {}
        self.loss.compute()
        for name, metric in self.metrics.items():
            result[name] = metric.compute()
        if show:
            self.show_compute(result)

    def get_epoc_batch_display(self):
        # print batch and epoch
        if self.current_step is None or self.current_epoch is None:
            out_str = ""
        else:
            out_str = self.batch_epoch_form.format(
                self.current_epoch, self.current_step
            )
        return out_str

    def show(self, result):
        out_str = self.get_epoc_batch_display()
        for metric_name, metric_value in result.items():
            out_str += self.format_single(metric_name, metric_value)

        # print losst
        out_str += self.format_single("loss", self.loss.compute())
        print(out_str, flush=True)

    def show_compute(self, result):
        if self.current_step is None:
            out_str = "Cumulative: \n"
        else:
            out_str = "\n[{}] Batch cumulative: \n".format(self.current_step)

        for metric_name, metric_value in result.items():
            out_str += self.format_compute(metric_name, metric_value)

        # print loss
        out_str += self.format_compute("loss", self.loss.compute())

        print(out_str, flush=True)

    # destructor
    def __del__(self):
        self.writer.flush()
        self.writer.close()


if __name__ == "__main__":
    # set device to GPU
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metrics = ModelMetrics(device=DEVICE, comment="moj_model_zanimiv")
    metrics.add_metric(
        "auroc",
        torchmetrics.AUROC(task="multiclass", num_classes=2, average="macro"),
    )
    metrics.add_metric(
        "accuracy", torchmetrics.Accuracy(task="multiclass", num_classes=2)
    )
    metrics.add_metric(
        "precision",
        torchmetrics.Precision(task="multiclass", num_classes=2, average="macro"),
    )

    # dummy trainings
    for i in range(10):
        metrics.reset()
        for j in range(100):
            # generate dummy data float32 with 2 classes
            y_pred = torch.rand(16, 2, dtype=torch.float32)
            y_true = torch.randint(0, 2, (16,), dtype=torch.int64)

            metrics.update(y_pred, y_true, 0.2, show=True, batch=j, epoch=i)

        metrics.compute(show=True)
        metrics.reset()
