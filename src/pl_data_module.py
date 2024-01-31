from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from .dataset import DrugResponseDataset


class DrugResponseDataModule(LightningDataModule):
    def __init__(
        self,
        cpd_embeddings_path,
        ccl_ge_path,
        drp_path,
        cpd_type="smiles",
        embed_tokens=False,
        train_split=0.8,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.cpd_embeddings_path = cpd_embeddings_path
        self.ccl_ge_path = ccl_ge_path
        self.drp_path = drp_path
        self.cpd_type = cpd_type
        self.embed_tokens = embed_tokens
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Initialize the dataset
        dataset = DrugResponseDataset(
            cpd_embeddings_path=self.cpd_embeddings_path,
            ccl_ge_path=self.ccl_ge_path,
            drp_path=self.drp_path,
            cpd_type=self.cpd_type,
            embed_tokens=self.embed_tokens,
        )

        # Split the dataset
        train_size = int(self.train_split * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=False,
        )
