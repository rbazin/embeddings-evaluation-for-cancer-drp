from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    DeviceStatsMonitor,
    StochasticWeightAveraging,
)
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler

import torch

import os
import datetime
import argparse

from src import DrugResponseLightningModule
from src import DrugResponseDataModule


def get_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Train a drug response model with PyTorch Lightning."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="smiles_model",
        help="Name of the model to train",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to a checkpoint to resume training from (optional)",
    )
    parser.add_argument(
        "--cpd_embeddings_path",
        type=str,
        default="data/smiles_train_embeddings_dict.joblib",
        help="Path to compound embeddings",
    )
    parser.add_argument(
        "--ccl_ge_path",
        type=str,
        default="data/ge_filtered_scaled.csv",
        help="Path to cell line gene expression data",
    )
    parser.add_argument(
        "--drp_path",
        type=str,
        default="data/drp_train.csv",
        help="Path to drug response data",
    )
    parser.add_argument(
        "--cpd_type", type=str, default="smiles", help="Type of compound representation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of CPU cores to use for dataloaders",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Wether to train with half precision or not",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Directory to save TensorBoard logs"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=256,
        help="Length of the compound sequence",
    )
    parser.add_argument(
        "--cpd_embedding_dim",
        type=int,
        default=768,
        help="Dimension of the compound embedding",
    )
    parser.add_argument(
        "--ccl_embedding_dim",
        type=int,
        default=6136,
        help="Dimension of the cell line embedding",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=1020, help="Dimension of the hidden layer"
    )
    parser.add_argument(
        "--transformer_heads", type=int, default=6, help="Number of transformer heads"
    )
    parser.add_argument(
        "--transformer_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=7924,
        help="Number of tokens in the vocabulary",
    )
    parser.add_argument(
        "--embed_tokens",
        action="store_true",
        help="Whether to use tokens as embeddings or not",
    )
    parser.add_argument(
        "--fingerprints",
        action="store_true",
        help="Whether the compound embeddings are fingerprints or not",
    )
    parser.add_argument(
        "--use_profiler",
        action="store_true",
        help="Whether to use the profiler or not",
    )
    parser.add_argument(
        "--profiler_dir",
        type=str,
        default="",
        help="Directory to save profiler logs",
    )
    parser.add_argument(
        "--tune_lr",
        action="store_true",
        help="Whether to look for the best learning rate before training or not",
    )
    parser.add_argument(
        "--use_swa",
        action="store_true",
        help="Whether to use stochastic weight averaging or not",
    )
    parser.add_argument(
        "--swa_lr",
        type=float,
        default=1e-4,
        help="Learning rate for stochastic weight averaging",
    )

    args = parser.parse_args()

    if args.use_profiler:
        assert (
            args.profiler_dir
        ), "Profiler directory must be specified when using the profiler"
        assert os.path.exists(args.profiler_dir), "Profiler directory must exist"

    assert  args.cpd_type in ["smiles", "selfies"], "Compound type must be either 'smiles' or 'selfies'"
    assert not (
        args.embed_tokens and args.fingerprints
    ), "Choose either tokens or fingerprints as embeddings, not both"

    return args


def main(args):
    """Main function to train the drug response model."""
    
    # Lightning Module
    drug_response_module = DrugResponseLightningModule(
        sequence_length=args.sequence_length,
        cpd_embedding_dim=args.cpd_embedding_dim,
        ccl_embedding_dim=args.ccl_embedding_dim,
        hidden_dim=args.hidden_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        learning_rate=args.learning_rate,
        fingerprints=args.fingerprints,
        embed_tokens=args.embed_tokens,
        vocab_size=args.vocab_size,
    )

    # Lightning Data Module
    drug_response_data_module = DrugResponseDataModule(
        cpd_embeddings_path=args.cpd_embeddings_path,
        ccl_ge_path=args.ccl_ge_path,
        drp_path=args.drp_path,
        cpd_type=args.cpd_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        embed_tokens=args.embed_tokens,
    )

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model_name_with_timestamp = f"{args.model_name}_{current_time}"
    checkpoint_dir_with_timestamp = os.path.join(
        args.checkpoint_dir, model_name_with_timestamp
    )
    log_dir_with_timestamp = os.path.join(args.log_dir, model_name_with_timestamp)

    # Callbacks
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir_with_timestamp,
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=15,
        verbose=True,
        mode="min",
    )
    callbacks.append(early_stopping_callback)

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor_callback)

    device_stats_monitor_callback = DeviceStatsMonitor(cpu_stats=True)
    callbacks.append(device_stats_monitor_callback)

    if args.use_swa:
        swa_callback = StochasticWeightAveraging(
            swa_lrs=args.swa_lr, annealing_epochs=5
        )
        callbacks.append(swa_callback)

    # Logger
    logger = TensorBoardLogger(log_dir_with_timestamp, name=args.model_name)

    # Performance Profiler
    profiler = (
        AdvancedProfiler(
            dirpath=args.profiler_dir,
            filename=f"{model_name_with_timestamp}_profiler",
        )
        if args.use_profiler
        else None
    )

    # Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        num_nodes=1,
        devices=-1,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="16-mixed" if args.half_precision else "32",
        callbacks=callbacks,
        logger=logger,
        inference_mode=True,
        profiler=profiler,
    )

    # WARNING: Doesn't work with DDP
    if args.tune_lr:
        tuner = Tuner(trainer)
        tuner.lr_find(drug_response_module, datamodule=drug_response_data_module)

    # Train the model
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Resuming training from checkpoint: {args.checkpoint_path}")
        trainer.fit(
            drug_response_module,
            drug_response_data_module,
            ckpt_path=args.checkpoint_path,
        )
    else:
        print("Starting training from scratch")
        trainer.fit(drug_response_module, datamodule=drug_response_data_module)
    
    # Finalize and save logs
    logger.finalize("success")


if __name__ == "__main__":
    args = get_args()
    main(args)
