from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
from .model import DrugResponseModel, DrugResponseModelTokens, DrugResponseModelMorganFingerprints

class DrugResponseLightningModule(LightningModule):
    def __init__(
        self,
        sequence_length=256,
        cpd_embedding_dim=768,
        ccl_embedding_dim=6136,
        hidden_dim=1020,
        transformer_heads=6,
        transformer_layers=6,
        learning_rate=1e-3,
        embed_tokens=False,
        vocab_size=7924,
        fingerprints=False,
        num_skip_layers=4,
        num_layers_before_skip=2,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if self.hparams.embed_tokens:
            self.model = DrugResponseModelTokens(
                cpd_sequence_length=self.hparams.sequence_length,
                cpd_embedding_dim=self.hparams.cpd_embedding_dim,
                ccl_embedding_dim=self.hparams.ccl_embedding_dim,
                hidden_dim=self.hparams.hidden_dim,
                transformer_heads=self.hparams.transformer_heads,
                transformer_layers=self.hparams.transformer_layers,
                vocab_size=self.hparams.vocab_size,
            )
        elif self.hparams.fingerprints:
            self.model = DrugResponseModelMorganFingerprints(
                cpd_embedding_dim=self.hparams.cpd_embedding_dim,
                ccl_embedding_dim=self.hparams.ccl_embedding_dim,
                hidden_dim=self.hparams.hidden_dim,
                num_skip_layers=self.hparams.num_skip_layers,
                num_layers_before_skip=self.hparams.num_layers_before_skip,
            )
        else:
            self.model = DrugResponseModel(
                cpd_sequence_length=self.hparams.sequence_length,
                cpd_embedding_dim=self.hparams.cpd_embedding_dim,
                ccl_embedding_dim=self.hparams.ccl_embedding_dim,
                hidden_dim=self.hparams.hidden_dim,
                transformer_heads=self.hparams.transformer_heads,
                transformer_layers=self.hparams.transformer_layers,
            )
        self.loss_fn = F.mse_loss

    def training_step(self, batch, batch_idx):
        cpd_emb = batch["cpd_embeddings"]
        ccl_emb = batch["ccl_ge_embeddings"]
        labels = batch["label"]
        outputs = self.model(cpd_emb, ccl_emb)
        loss = self.loss_fn(outputs, labels)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            # rank_zero_only=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        cpd_emb = batch["cpd_embeddings"]
        ccl_emb = batch["ccl_ge_embeddings"]
        labels = batch["label"]
        outputs = self.model(cpd_emb, ccl_emb)
        loss = self.loss_fn(outputs, labels)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            # rank_zero_only=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
