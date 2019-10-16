from keras import callbacks, backend as K


class ObserveMetrics(callbacks.Callback):
    def __init__(self, run_obj, prefix):
        super().__init__()
        self._run = run_obj
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.observe_metrics(epoch=epoch, logs=logs)

    def observe_metrics(self, epoch, logs):
        for k, v in logs.items():
            self._run.log_scalar(f'{self.prefix}_{k}', float(v), step=epoch)
        self._run.log_scalar(f"{self.prefix}_lr", float(K.get_value(self.model.optimizer.lr)), step=epoch)
