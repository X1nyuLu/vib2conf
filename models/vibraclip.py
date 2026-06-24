"""
VibraCLIP baseline adapted to Vib2Conf framework.

Original: DimeNet++ (molecule) + MLP (spectra) contrastive learning.
Architecture params are hardcoded to match the original paper defaults.
Supports IR-only and IR+Raman variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import DimeNetPlusPlus

from . import register_model


def _safe_spec(data, *keys):
    """Return the first non-None spectrum tensor from *keys*, or None."""
    for k in keys:
        try:
            v = getattr(data, k)
            if v is not None:
                return v
        except AttributeError:
            pass
    return None



# ============================================================
#  Building blocks (preserved from original VibraCLIP)
# ============================================================

def _get_act(name):
    return {"relu": nn.ReLU(), "elu": nn.ELU(), "leakyrelu": nn.LeakyReLU(),
            "softplus": nn.Softplus(), "tanh": nn.Tanh()}[name]


class GNNEncoder(nn.Module):
    """DimeNet++ encoder."""

    def __init__(self, hidden_channels=128, out_channels=181, num_blocks=4,
                 int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
                 num_spherical=7, num_radial=6, cutoff=5.0, max_num_neighbors=32,
                 envelope_exponent=5, num_before_skip=1, num_after_skip=2,
                 num_output_layers=3):
        super().__init__()
        self.graph_conv = DimeNetPlusPlus(
            hidden_channels=hidden_channels, out_channels=out_channels,
            num_blocks=num_blocks, int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size, out_emb_channels=out_emb_channels,
            num_spherical=num_spherical, num_radial=num_radial, cutoff=cutoff,
            max_num_neighbors=max_num_neighbors, envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip, num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
        )

    def forward(self, z, pos, batch):
        return self.graph_conv(z, pos, batch)


class SpectraEncoder(nn.Module):
    """MLP encoder for vibrational spectra."""

    def __init__(self, input_dim=1024, hidden_dim=1262, n_layers=1,
                 out_features=497, act_fun="elu", batch_norm=True):
        super().__init__()
        act = _get_act(act_fun)
        inter_dim = np.linspace(hidden_dim, out_features, n_layers + 1, dtype=int)[:-1]
        inter_dim = np.concatenate(([input_dim], inter_dim))

        layers = []
        for in_s, out_s in zip(inter_dim[:-1], inter_dim[1:]):
            layers.append(nn.Linear(in_s, out_s))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_s))
            layers.append(act)
        self.hidden = nn.Sequential(*layers)
        self.lin_out = nn.Linear(inter_dim[-1], out_features)

    def forward(self, x):
        return self.lin_out(self.hidden(x))


class ProjectionHead(nn.Module):
    """Residual projection head."""

    def __init__(self, embedding_dim, projection_dim, dropout=False,
                 p_dropout=0.0, layer_norm=True, bias=False):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim, bias=bias)
        self.fc = nn.Linear(projection_dim, projection_dim, bias=bias)
        self.gelu = nn.GELU()
        self.use_dropout = dropout
        self.use_layer_norm = layer_norm
        if dropout:
            self.dropout_layer = nn.Dropout(p_dropout)
        if layer_norm:
            self.layer_norm_out = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        if self.use_dropout:
            x = self.dropout_layer(x)
        x += projected
        if self.use_layer_norm:
            x = self.layer_norm_out(x)
        return x


# ============================================================
#  Loss functions (soft-target contrastive)
# ============================================================

def _loss_single(graph_emb, spec_emb, temperature, log_softmax_fn):
    """Single-modality loss: graph <-> spectra alignment with soft targets."""
    logits = (spec_emb @ graph_emb.T) / temperature
    graph_sim = graph_emb @ graph_emb.T
    spec_sim = spec_emb @ spec_emb.T
    targets = F.softmax((graph_sim + spec_sim) / 2.0 * temperature, dim=-1)
    graph_loss = (-targets.T * log_softmax_fn(logits.T)).sum(1)
    spec_loss = (-targets * log_softmax_fn(logits)).sum(1)
    return (graph_loss + spec_loss).mean() / 2.0


def _loss_multi(graph_embeddings, ir_spectra_embeddings, raman_spectra_embeddings, temperature, log_softmax_fn):

    """Loss function that aligns Graph, IR and Ramam embeddings with IR-Raman alignment. AllPairs."""
    # Compute logits for each pairwise combination
    logits_ir_graph = (
        ir_spectra_embeddings @ graph_embeddings.T
    ) / temperature
    logits_raman_graph = (
        raman_spectra_embeddings @ graph_embeddings.T
    ) / temperature
    logits_ir_raman = (
        ir_spectra_embeddings @ raman_spectra_embeddings.T
    ) / temperature

    # Similarities within each modality and between IR-Raman
    graph_similarities = graph_embeddings @ graph_embeddings.T
    ir_spectra_similarities = (
        ir_spectra_embeddings @ ir_spectra_embeddings.T
    )
    raman_spectra_similarities = (
        raman_spectra_embeddings @ raman_spectra_embeddings.T
    )
    ir_raman_similarities = (
        ir_spectra_embeddings @ raman_spectra_embeddings.T
    )

    # Targets: average across all modality combinations
    targets = F.softmax(
        (
            graph_similarities
            + ir_spectra_similarities
            + raman_spectra_similarities
            + ir_raman_similarities
        )
        / 4.0
        * temperature,
        dim=-1,
    )

    # Compute loss terms for each modality pair
    graph_loss_ir = (-targets.T * log_softmax_fn(logits_ir_graph.T)).sum(
        1
    )
    graph_loss_raman = (
        -targets.T * log_softmax_fn(logits_raman_graph.T)
    ).sum(1)
    ir_spectra_loss = (-targets * log_softmax_fn(logits_ir_graph)).sum(1)
    raman_spectra_loss = (
        -targets * log_softmax_fn(logits_raman_graph)
    ).sum(1)
    ir_raman_loss = (-targets * log_softmax_fn(logits_ir_raman)).sum(1)

    # Average the losses across all modalities
    avg_loss = (
        graph_loss_ir
        + graph_loss_raman
        + ir_spectra_loss
        + raman_spectra_loss
        + ir_raman_loss
    ) / 5.0

    return avg_loss.mean()

# ============================================================
#  VibraCLIP — IR only
# ============================================================

@register_model
def vibraclip(**kwargs):
    return VibraCLIP(**kwargs)


class VibraCLIP(nn.Module):
    """VibraCLIP with a single spectral modality (IR, Raman, or either)."""

    def __init__(self, **kwargs):
        super().__init__()
        self.temperature = kwargs.get('temperature', 141)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.gnn_encoder = GNNEncoder(
            hidden_channels=kwargs.get('g_encoder_hidden_channels', 128),
            out_channels=kwargs.get('g_encoder_out_channels', 181),
            num_blocks=kwargs.get('g_encoder_num_blocks', 4),
            int_emb_size=kwargs.get('g_encoder_int_emb_size', 64),
            basis_emb_size=kwargs.get('g_encoder_basis_emb_size', 8),
            out_emb_channels=kwargs.get('g_encoder_out_emb_channels', 256),
            num_spherical=kwargs.get('g_encoder_num_spherical', 7),
            num_radial=kwargs.get('g_encoder_num_radial', 6),
            cutoff=kwargs.get('g_encoder_cutoff', 5.0),
            max_num_neighbors=kwargs.get('g_encoder_max_num_neighbors', 32),
            envelope_exponent=kwargs.get('g_encoder_envelope_exponent', 5),
            num_before_skip=kwargs.get('g_encoder_num_before_skip', 1),
            num_after_skip=kwargs.get('g_encoder_num_after_skip', 2),
            num_output_layers=kwargs.get('g_encoder_num_output_layers', 3),
        )

        g_out = kwargs.get('g_encoder_out_channels', 181)
        spec_out = kwargs.get('spectra_encoder_out_features', 497)
        proj_dim = kwargs.get('projection_latent_dim', 861)

        self.spectra_encoder = SpectraEncoder(
            input_dim=kwargs.get('spectra_encoder_input_dim', 1024),
            hidden_dim=kwargs.get('spectra_encoder_hidden_dim', 1262),
            n_layers=kwargs.get('spectra_encoder_n_layers', 1),
            out_features=spec_out,
            act_fun=kwargs.get('spectra_encoder_act_fun', 'elu'),
            batch_norm=kwargs.get('spectra_encoder_batch_norm', True),
        )

        self.graph_proj = ProjectionHead(
            g_out, proj_dim,
            dropout=kwargs.get('projection_dropout', False),
            p_dropout=kwargs.get('projection_p_dropout', 0.0),
            layer_norm=kwargs.get('projection_layer_norm', True),
            bias=kwargs.get('projection_bias', False),
        )
        self.spec_proj = ProjectionHead(
            spec_out, proj_dim,
            dropout=kwargs.get('projection_dropout', False),
            p_dropout=kwargs.get('projection_p_dropout', 0.0),
            layer_norm=kwargs.get('projection_layer_norm', True),
            bias=kwargs.get('projection_bias', False),
        )

    def forward(self, inputs, return_loss=True, return_proj_output=False):
        z = inputs.x.squeeze()
        pos = inputs.pos
        batch = inputs.batch
        n_graphs = int(batch.max().item()) + 1

        graph_feat = self.gnn_encoder(z, pos, batch)

        spec = _safe_spec(inputs, "ir", "raman")
        spec_feat = self.spectra_encoder(spec.view(n_graphs, -1))

        graph_emb = self.graph_proj(graph_feat)
        spec_emb = self.spec_proj(spec_feat)

        result = {}
        if return_loss:
            loss = _loss_single(graph_emb, spec_emb, self.temperature, self.log_softmax)
            result['loss'] = loss
            result['cl_loss'] = loss
        if return_proj_output:
            result['molecular_proj_output'] = graph_emb
            result['spectral_proj_output'] = spec_emb
        return result


# ============================================================
#  VibraCLIP Multi — IR + Raman
# ============================================================

@register_model
def vibraclip_dual(**kwargs):
    return VibraCLIPDual(**kwargs)


class VibraCLIPDual(nn.Module):
    """VibraCLIP with dual spectral modalities, optional allpairs loss."""

    def __init__(self, **kwargs):
        super().__init__()
        self.temperature = kwargs.get('temperature', 141)
        self.loss_allpairs = kwargs.get('loss_allpairs', False)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        g_out = kwargs.get('g_encoder_out_channels', 181)
        spec_out = kwargs.get('spectra_encoder_out_features', 497)
        proj_dim = kwargs.get('projection_latent_dim', 861)

        self.gnn_encoder = GNNEncoder(
            hidden_channels=kwargs.get('g_encoder_hidden_channels', 128),
            out_channels=g_out,
            num_blocks=kwargs.get('g_encoder_num_blocks', 4),
            int_emb_size=kwargs.get('g_encoder_int_emb_size', 64),
            basis_emb_size=kwargs.get('g_encoder_basis_emb_size', 8),
            out_emb_channels=kwargs.get('g_encoder_out_emb_channels', 256),
            num_spherical=kwargs.get('g_encoder_num_spherical', 7),
            num_radial=kwargs.get('g_encoder_num_radial', 6),
            cutoff=kwargs.get('g_encoder_cutoff', 5.0),
            max_num_neighbors=kwargs.get('g_encoder_max_num_neighbors', 32),
            envelope_exponent=kwargs.get('g_encoder_envelope_exponent', 5),
            num_before_skip=kwargs.get('g_encoder_num_before_skip', 1),
            num_after_skip=kwargs.get('g_encoder_num_after_skip', 2),
            num_output_layers=kwargs.get('g_encoder_num_output_layers', 3),
        )

        spectra_enc_kwargs = dict(
            input_dim=kwargs.get('spectra_encoder_input_dim', 1024),
            hidden_dim=kwargs.get('spectra_encoder_hidden_dim', 1262),
            n_layers=kwargs.get('spectra_encoder_n_layers', 1),
            out_features=spec_out,
            act_fun=kwargs.get('spectra_encoder_act_fun', 'elu'),
            batch_norm=kwargs.get('spectra_encoder_batch_norm', True),
        )
        proj_kwargs = dict(
            dropout=kwargs.get('projection_dropout', False),
            p_dropout=kwargs.get('projection_p_dropout', 0.0),
            layer_norm=kwargs.get('projection_layer_norm', True),
            bias=kwargs.get('projection_bias', False),
        )

        self.ir_encoder = SpectraEncoder(**spectra_enc_kwargs)
        self.raman_encoder = SpectraEncoder(**spectra_enc_kwargs)

        self.graph_proj = ProjectionHead(g_out, proj_dim, **proj_kwargs)
        self.ir_proj = ProjectionHead(spec_out, proj_dim, **proj_kwargs)
        self.raman_proj = ProjectionHead(spec_out, proj_dim, **proj_kwargs)

    def forward(self, inputs, return_loss=True, return_proj_output=False):
        z = inputs.x.squeeze()
        pos = inputs.pos
        batch = inputs.batch
        n_graphs = int(batch.max().item()) + 1

        graph_feat = self.gnn_encoder(z, pos, batch)

        ir_feat = self.ir_encoder(_safe_spec(inputs, 'ir').view(n_graphs, -1))
        raman_feat = self.raman_encoder(_safe_spec(inputs, 'raman').view(n_graphs, -1))

        graph_emb = self.graph_proj(graph_feat)
        ir_emb = self.ir_proj(ir_feat)
        raman_emb = self.raman_proj(raman_feat)

        result = {}
        if return_loss:
            loss = _loss_multi(graph_emb, ir_emb, raman_emb,
                               self.temperature, self.log_softmax)
            result['loss'] = loss
            result['cl_loss'] = loss
        if return_proj_output:
            result['molecular_proj_output'] = graph_emb
            result['spectral_proj_output'] = (ir_emb + raman_emb) / 2.0
            result['ir_proj_output'] = ir_emb
            result['raman_proj_output'] = raman_emb
        return result
