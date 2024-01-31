from torch.utils.data import Dataset
import pandas as pd
import torch
import joblib


class DrugResponseDataset(Dataset):
    def __init__(
        self,
        cpd_embeddings_path,
        ccl_ge_path,
        drp_path,
        cpd_type="smiles",
        embed_tokens=False,
        transform=None,
    ):
        self.cpd_embeddings = joblib.load(cpd_embeddings_path)
        self.ccl_ge = pd.read_csv(ccl_ge_path, index_col=0, header=0)
        self.drp = pd.read_csv(drp_path, header=0)

        # test the validity of the datasets
        assert len(self.cpd_embeddings) > 0, "Compound embeddings are empty"
        assert len(self.ccl_ge) > 0, "Cell line gene expression is empty"
        assert len(self.drp) > 0, "Drug response data is empty"
        
        unique_cpds = set(self.drp[cpd_type].unique())
        embedding_keys = set(self.cpd_embeddings.keys())
        assert unique_cpds.issubset(embedding_keys), "Not all unique compounds are in the embedding list"

        self.cpd_type = cpd_type
        self.transform = transform
        self.embed_tokens = embed_tokens

    def __len__(self):
        return len(self.drp)

    def __getitem__(self, idx):
        ccl_name = self.drp.iloc[idx]["ccl_name"]
        ccl_ge_embeddings = self.ccl_ge.loc[ccl_name].values

        cpd_name = self.drp.iloc[idx][self.cpd_type]
        cpd_embeddings = self.cpd_embeddings[cpd_name]

        if self.transform:
            cpd_embeddings = self.transform(cpd_embeddings)
            ccl_ge_embeddings = self.transform(ccl_ge_embeddings)

        label = self.drp.iloc[idx]["area_under_curve_scaled"]

        return {
            "cpd_name": cpd_name,
            "ccl_name": ccl_name,
            "cpd_embeddings": torch.tensor(
                cpd_embeddings,
                dtype=torch.float32 if not self.embed_tokens else torch.int32,
            ),
            "ccl_ge_embeddings": torch.tensor(ccl_ge_embeddings, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
        }
