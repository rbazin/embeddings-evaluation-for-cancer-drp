import torch
from torch import nn
from torch.nn import functional as F


class DrugResponseModel(nn.Module):
    def __init__(
        self,
        cpd_sequence_length=256,
        cpd_embedding_dim=768,
        ccl_embedding_dim=6136,
        hidden_dim=1020,
        transformer_heads=6,
        transformer_layers=6,
    ):
        super().__init__()

        self.positional_embedding = nn.Parameter(
            torch.zeros(1, cpd_sequence_length, cpd_embedding_dim)
        )
        nn.init.xavier_uniform_(self.positional_embedding)

        self.ccl_to_cpd_projection = nn.Linear(ccl_embedding_dim, cpd_embedding_dim)
        self.combined_to_hidden_projection = nn.Linear(cpd_embedding_dim, hidden_dim)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            activation=F.leaky_relu,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        self.regression_head = nn.Linear(hidden_dim, 1)

    def forward(self, cpd_embed_seq, ccl_embed):
        batch_size = cpd_embed_seq.size(0)

        ccl_embed = self.ccl_to_cpd_projection(ccl_embed)
        ccl_embed = F.leaky_relu(ccl_embed)

        ccl_embed = ccl_embed.unsqueeze(1).expand(-1, cpd_embed_seq.size(1), -1)

        cpd_embed_seq = cpd_embed_seq + self.positional_embedding.expand(
            batch_size, -1, -1
        )

        x = cpd_embed_seq + ccl_embed
        del cpd_embed_seq, ccl_embed

        x = self.combined_to_hidden_projection(x)
        x = F.leaky_relu(x)

        x = self.transformer_encoder(x)

        x = x.mean(dim=1)
        x = F.leaky_relu(x)

        x = self.regression_head(x).squeeze(-1)

        return x


class DrugResponseModelMorganFingerprints(nn.Module):
    def __init__(
        self,
        cpd_embedding_dim=2048,
        ccl_embedding_dim=6136,
        hidden_dim=2040,
        num_skip_layers=4,
        num_layers_before_skip=2,
    ):
        super().__init__()

        self.ccl_projection = nn.Linear(ccl_embedding_dim, hidden_dim)
        self.cpd_projection = nn.Linear(cpd_embedding_dim, hidden_dim)
        self.combined_to_hidden_projection = nn.Linear(2 * hidden_dim, hidden_dim)

        layers = []
        for _ in range(num_skip_layers):
            layers.append(LinearSkipLayer(hidden_dim, num_layers_before_skip))

        self.deepset = nn.Sequential(*layers)

        self.regression_head = nn.Linear(hidden_dim, 1)

    def forward(self, cpd_embed, ccl_embed):
        ccl_embed = self.ccl_projection(ccl_embed)
        ccl_embed = F.leaky_relu(ccl_embed)

        cpd_embed = self.cpd_projection(cpd_embed)
        cpd_embed = F.leaky_relu(cpd_embed)

        x = torch.cat([ccl_embed, cpd_embed], dim=1)
        del cpd_embed, ccl_embed

        x = self.combined_to_hidden_projection(x)
        x = F.leaky_relu(x)

        x = self.deepset(x)

        x = self.regression_head(x).squeeze(-1)

        return x


class LinearSkipLayer(nn.Module):
    def __init__(self, hidden_dim=2040, num_layers_before_skip=1) -> None:
        super().__init__()

        layers = []
        for _ in range(num_layers_before_skip):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU())

        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.linear(x)


class DrugResponseModelTokens(nn.Module):
    def __init__(
        self,
        cpd_sequence_length=256,
        cpd_embedding_dim=768,
        ccl_embedding_dim=6136,
        hidden_dim=1020,
        transformer_heads=6,
        transformer_layers=6,
        vocab_size=7924,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, cpd_embedding_dim)
        self.drp_model = DrugResponseModel(
            cpd_sequence_length=cpd_sequence_length,
            cpd_embedding_dim=cpd_embedding_dim,
            ccl_embedding_dim=ccl_embedding_dim,
            hidden_dim=hidden_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
        )

    def forward(self, cpd_embed_seq, ccl_embed):
        cpd_embed_seq = self.token_embedding(cpd_embed_seq)
        cpd_embed_seq = F.leaky_relu(cpd_embed_seq)
        return self.drp_model(cpd_embed_seq, ccl_embed)


class DrugResponseModelLegacy(nn.Module):
    def __init__(
        self,
        cpd_sequence_length=256,
        cpd_embedding_dim=768,
        ccl_embedding_dim=6136,
        hidden_dim=1020,
        transformer_heads=4,
        transformer_layers=2,
    ):
        super().__init__()

        self.positional_embedding = nn.Parameter(
            torch.zeros(1, cpd_sequence_length, cpd_embedding_dim)
        )
        nn.init.xavier_uniform_(self.positional_embedding)

        self.ccl_to_cpd_projection = nn.Linear(ccl_embedding_dim, cpd_embedding_dim)
        self.combined_to_hidden_projection = nn.Linear(cpd_embedding_dim, hidden_dim)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=transformer_layers,
        )

        self.regression_head = nn.Linear(hidden_dim, 1)

    def forward(self, cpd_embed_seq, ccl_embed):
        batch_size = cpd_embed_seq.size(0)

        ccl_embed = (
            self.ccl_to_cpd_projection(ccl_embed)
            .unsqueeze(1)
            .expand(-1, cpd_embed_seq.size(1), -1)
        )

        cpd_embed_seq = cpd_embed_seq + self.positional_embedding.expand(
            batch_size, -1, -1
        )

        x = cpd_embed_seq + ccl_embed
        del cpd_embed_seq, ccl_embed

        x = self.combined_to_hidden_projection(x)

        x = self.transformer_encoder(x)

        x = x.mean(dim=1)

        x = self.regression_head(x).squeeze(-1)

        return x
