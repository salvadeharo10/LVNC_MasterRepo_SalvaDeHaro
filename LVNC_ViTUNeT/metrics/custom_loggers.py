import csv
from pytorch_lightning.loggers.logger import Logger  # <- import actualizado

class CsvLogger(Logger):
    def __init__(self, train_csv_path, val_csv_path, train_header, val_header):
        super().__init__()
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path

        with open(self.train_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(train_header)
        self.train_header = train_header

        with open(self.val_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(val_header)
        self.val_header = val_header

    def log_metrics(self, metrics, step):
        if any(key in metrics for key in ["train/loss", "loss_epoch", "loss"]):
            row = []
            for k in self.train_header:
                if k == "epoch":
                    row.append(metrics.get('epoch', ''))
                elif k == "train/loss":
                    row.append(metrics.get('loss_epoch', metrics.get('train/loss', '')))
                else:
                    row.append(metrics.get(k, ""))
            filename = self.train_csv_path
        elif any(key in metrics for key in ["val/loss", "validation/loss"]):
            row = [metrics.get(k, "") for k in self.val_header]
            filename = self.val_csv_path
        else:
            return
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_hyperparams(self, params):
        pass  # Puedes implementar si necesitas registrar hiperparámetros

    def finalize(self, status):
        pass  # Puedes limpiar recursos aquí si quieres

    @property
    def name(self):
        return "csvlogger"

    @property
    def version(self):
        return "1.0"

    @property
    def experiment(self):
        return None  # No hay objeto "experiment" como en TensorBoard, así que devolvemos None
