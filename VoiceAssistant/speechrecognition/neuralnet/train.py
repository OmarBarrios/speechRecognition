import os
import ast
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from model import SpeechRecognition
from dataset import Data, collate_fn_padd
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Speech Recognition Module Class

class SpeechRecModule(LightningModule):
    def __init__(self, model, args):
        """
        Initializes the SpeechRecModule with a model and additional arguments.

        Args:
            model: PyTorch model for speech recognition.
            args: Additional arguments for customization.
        """
        super(SpeechRecModule, self).__init__()
        self.model = model
        self.args = args

    def forward(self, x, hidden):
        """
        Performs the forward pass of the model.

        Args:
            x: Input data.
            hidden: Hidden state.

        Returns:
            Model predictions.
        """
        return self.model(x, hidden)

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training.

        Returns:
            Optimizer and scheduler.
        """
        # Define optimizer and scheduler (source: [2](https://lightning.ai/docs/pytorch/stable//common/lightning_module.html))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return [optimizer], [scheduler]

    def step(self, batch):
        """
        Performs a single training step, including forward pass, calculating loss, and returning the loss value.

        Args:
            batch: Training batch.

        Returns:
            Loss value.
        """
        x, y = batch
        predictions = self.model(x)
        loss = F.cross_entropy(predictions, y)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Performs a training step and returns the loss and logs.

        Args:
            batch: Training batch.
            batch_idx: Index of the batch.

        Returns:
            Dictionary with 'loss' key.
        """
        loss = self.step(batch)
        self.log('train_loss', loss)
        return {'loss': loss}

    def train_dataloader(self):
        """
        Returns the dataloader for the training dataset.

        Returns:
            Training dataloader.
        """
        d_params = Data.parameters
        d_params.update(self.args.dparams_override)
        train_dataset = Data(json_path=self.args.train_file, **d_params)
        return DataLoader(dataset=train_dataset,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.data_workers,
                          pin_memory=True,
                          collate_fn=collate_fn_padd)

    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step and returns the validation loss.

        Args:
            batch: Validation batch.
            batch_idx: Index of the batch.

        Returns:
            Validation loss.
        """
        loss = self.step(batch)
        return loss

    def on_validation_epoch_end(self):
        """
        Called at the end of each validation epoch, adjusts the learning rate scheduler and logs the validation loss.
        """
        # Adjust learning rate scheduler if needed
        pass

    def val_dataloader(self):
        """
        Returns the dataloader for the validation dataset.

        Returns:
            Validation dataloader.
        """
        d_params = Data.parameters
        d_params.update(self.args.dparams_override)
        test_dataset = Data(json_path=self.args.valid_file, **d_params, valid=True)
        return DataLoader(dataset=test_dataset, batch_size=self.args.batch_size, shuffle=False)

def checkpoint_callback(args):
    return ModelCheckpoint(
        dirpath=args.save_model_path,
        filename="model-{epoch:02d}-{val_loss:.2f}",  # Cambia según tus necesidades
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

def main(args):
    h_params = SpeechRecognition.hyper_parameters
    h_params.update(args.hparams_override)
    model = SpeechRecognition(**h_params)

    if args.load_model_from:
        speech_module = SpeechRecModule.load_from_checkpoint(args.load_model_from, model=model, args=args)
    else:
        speech_module = SpeechRecModule(model, args)

    logger = TensorBoardLogger(args.logdir, name='speech_recognition')

    trainer = Trainer(
        max_epochs=args.epochs,
        num_nodes=args.nodes,
        logger=logger,
        gradient_clip_val=1.0,
        val_check_interval=args.valid_every,
        callbacks=[checkpoint_callback(args)]
    )
    trainer.fit(speech_module)

if __name__ == "__main__":
    parser = ArgumentParser()
    # distributed training setup
    parser.add_argument('-n', '--nodes', default=1, type=int, help='number of data loading workers')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--data_workers', default=0, type=int,
                        help='n trabajadores de carga de datos, por defecto 0 = solo proceso principal.')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str,
                        help='qué backend distribuido usar. Por defecto, ddp.')

    # train and valid
    parser.add_argument('--train_file', default=None, required=True, type=str,
                        help='Archivo JSON para cargar datos de entrenamiento.')
    parser.add_argument('--valid_file', default=None, required=True, type=str,
                        help='Archivo JSON para cargar datos de prueba.')
    parser.add_argument('--valid_every', default=1000, required=False, type=int,
                        help='Validar después de cada N iteraciones.')

    # dir and path for models and logs
    parser.add_argument('--save_model_path', default=None, required=True, type=str,
                        help='Ruta para guardar el modelo.')
    parser.add_argument('--load_model_from', default=None, required=False, type=str,
                        help='Ruta para cargar un modelo preentrenado y continuar con el entrenamiento.')
    parser.add_argument('--resume_from_checkpoint', default=None, required=False, type=str,
                        help='Verifica la ruta para reanudar desde.')
    parser.add_argument('--logdir', default='tb_logs', required=False, type=str,
                        help='Ruta para guardar los registros (logs).')
    
    # general
    parser.add_argument('--epochs', default=10, type=int, help='Número total de epochs a ejecutar.')
    parser.add_argument('--batch_size', default=64, type=int, help='Tamaño del lote (batch).')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='Tasa de aprendizaje.')
    parser.add_argument('--pct_start', default=0.3, type=float, help='Porcentaje de la fase de crecimiento en un ciclo.')
    parser.add_argument('--div_factor', default=100, type=int, help='Factor de división para un ciclo.')
    parser.add_argument("--hparams_override", default="{}", type=str, required=False,
		help='Anular los hiperparámetros, deben estar en forma de diccionario. Es decir: {"attention_layers": 16 }')
    parser.add_argument("--dparams_override", default="{}", type=str, required=False,
		help='Anular los parámetros de datos, deben estar en forma de diccionario. Es decir: {"sample_rate": 8000 }')

    args = parser.parse_args()
    args.hparams_override = ast.literal_eval(args.hparams_override)
    args.dparams_override = ast.literal_eval(args.dparams_override)


    if args.save_model_path:
       if not os.path.isdir(os.path.dirname(args.save_model_path)):
           raise Exception("the directory for path {} does not exist".format(args.save_model_path))

    main(args)
