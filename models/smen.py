# -*- coding: utf-8 -*-
"""
SMEN baseline adapted to Vib2Conf framework.

Original: EGNN (molecule) + ViT (spectra) contrastive learning.
Rewritten to plug into the @register_model / spec2conf_base pattern.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter

from . import register_model


# ============================================================
#  EGNN building blocks (preserved from original SMEN)
# ============================================================

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL(nn.Module):
    """Edge / node / coord message passing layer."""

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
                 nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True,
                 coords_weight=1.0, attention=False, norm_diff=False,
                 tanh=False):
        super().__init__()
        self.recurrent = recurrent
        self.attention = attention
        self.tanh = tanh
        self.norm_diff = norm_diff
        self.coords_weight = coords_weight

        input_edge_nf = input_nf * 2 + 1  # +1 for radial distance
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1),
        )

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            out = out * self.att_mlp(out)
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord = coord + agg * self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def forward(self, h, edge_index, coord, edge_attr=None,
                node_attr=None, edge_mask=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        if edge_mask is not None:
            edge_feat = edge_feat * edge_mask
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, _ = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_feat


class E_GCL_mask(E_GCL):
    """E_GCL with node_mask support (for padded batches)."""

    def forward(self, h, edge_index, coord, node_mask, edge_mask,
                edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        edge_feat = edge_feat * edge_mask
        h, _ = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


class EGNN(nn.Module):
    """Equivariant Graph Neural Network (SMEN version).

    Expects fully-connected edges within each molecule, with node/edge masks
    for variable-size padding.
    """

    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, n_layers=4,
                 coords_weight=1.0, attention=False, node_attr=1,
                 output_size=512):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.node_attr = node_attr
        self.output_size = output_size

        self.embedding = nn.Linear(in_node_nf, hidden_nf)

        n_node_attr = in_node_nf if node_attr else 0
        for i in range(n_layers):
            self.add_module(
                f"gcl_{i}",
                E_GCL_mask(
                    hidden_nf, hidden_nf, hidden_nf,
                    edges_in_d=in_edge_nf,
                    nodes_att_dim=n_node_attr,
                    act_fn=nn.SiLU(),
                    recurrent=True,
                    coords_weight=coords_weight,
                    attention=attention,
                ),
            )

        self.node_dec = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, output_size),
        )

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask,
                n_nodes):
        h = self.embedding(h0)
        for i in range(self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules[f"gcl_{i}"](
                    h, edges, x, node_mask, edge_mask,
                    edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes,
                )
            else:
                h, _, _ = self._modules[f"gcl_{i}"](
                    h, edges, x, node_mask, edge_mask,
                    edge_attr=edge_attr, node_attr=None, n_nodes=n_nodes,
                )
        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred


# ============================================================
#  ViT building blocks (preserved from original SMEN)
# ============================================================

class Transpose(nn.Module):
    def __init__(self, d0, d1):
        super().__init__()
        self.d0, self.d1 = d0, d1

    def forward(self, x):
        return x.transpose(self.d0, self.d1)


def _attention(q, k, v, mask=None):
    B = q.shape[0]
    scale = q.shape[2] ** 0.5
    att = torch.bmm(q, k.transpose(1, 2)) / scale
    if mask is not None:
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        att = att.masked_fill(mask == 0, float("-inf"))
    att = F.softmax(att, dim=2)
    return torch.bmm(att, v)


class Head(nn.Module):
    def __init__(self, h_dim, head_out_dim):
        super().__init__()
        self.q_lin = nn.Linear(h_dim, head_out_dim, bias=False)
        self.k_lin = nn.Linear(h_dim, head_out_dim, bias=False)
        self.v_lin = nn.Linear(h_dim, head_out_dim, bias=False)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None:
            k = q
        if v is None:
            v = k
        return _attention(self.q_lin(q), self.k_lin(k), self.v_lin(v), mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, h_dim, num_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(h_dim, h_dim // num_heads) for _ in range(num_heads)]
        )
        self.linear = nn.Linear((h_dim // num_heads) * num_heads, h_dim)

    def forward(self, q, k=None, v=None, mask=None):
        x = torch.cat([h(q, k, v, mask=mask) for h in self.heads], dim=-1)
        return self.linear(x)


class ViTransformerEncoderLayer(nn.Module):
    def __init__(self, h_dim, num_heads, d_ff=2048, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(h_dim)
        self.mha = MultiHeadAttention(h_dim, num_heads)
        self.norm2 = nn.LayerNorm(h_dim)
        self.ffn = nn.Sequential(
            nn.Linear(h_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, h_dim),
        )

    def forward(self, x, mask=None):
        x = self.mha(self.norm1(x), mask=mask) + x
        x = self.ffn(self.norm2(x)) + x
        return x


class ViTransformerEncoder(nn.Module):
    def __init__(self, num_layers, h_dim, num_heads, d_ff=2048,
                 max_time_steps=None, use_clf_token=False,
                 dropout=0.0, dropout_emb=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            ViTransformerEncoderLayer(h_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.pos_emb = nn.Embedding(max_time_steps, h_dim)
        self.use_clf_token = use_clf_token
        if use_clf_token:
            self.clf_token = nn.Parameter(torch.randn(1, h_dim))
        self.dropout_emb = nn.Dropout(dropout_emb)

    def forward(self, x, mask=None):
        if self.use_clf_token:
            clf = self.clf_token.expand(x.size(0), -1, -1)
            x = torch.cat([clf, x], dim=1)
        B, T, _ = x.shape
        x = x + self.pos_emb.weight[:T].unsqueeze(0)
        x = self.dropout_emb(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class ViT(nn.Module):
    def __init__(self, patch_size, num_layers, h_dim, num_heads, output_size,
                 d_ff=2048, max_time_steps=None, use_clf_token=True,
                 dropout=0.0, dropout_emb=0.0):
        super().__init__()
        self.proc = nn.Sequential(
            nn.Unfold((1, patch_size), stride=(1, patch_size)),
            Transpose(1, 2),
            nn.Linear(patch_size, h_dim),
        )
        self.enc = ViTransformerEncoder(
            num_layers, h_dim, num_heads, d_ff=d_ff,
            max_time_steps=max_time_steps, use_clf_token=use_clf_token,
            dropout=dropout, dropout_emb=dropout_emb,
        )
        self.mlp = nn.Linear(h_dim, output_size)
        self._use_clf = use_clf_token

    def forward(self, x):
        x = self.proc(x)
        x = self.enc(x)
        x = x[:, 0] if self._use_clf else x.mean(dim=1)
        return self.mlp(x)


# ============================================================
#  SMEN → Vib2Conf wrapper
# ============================================================

class SMEN(nn.Module):
    """SMEN baseline: EGNN (molecule) + ViT (spectra), contrastive learning.

    Hyperparameters default to the best SMEN config on VB-Confs Raman.
    """

    def __init__(
        self,
        # --- EGNN ---
        in_node_nf=15,
        hidden_nf=256,
        n_layers=5,
        coords_weight=1.0,
        attention=True,
        mol_output_size=512,
        # --- ViT ---
        vit_patch_size=7,
        vit_num_layers=5,
        vit_h_dim=512,
        vit_num_heads=7,
        vit_d_ff=1024,
        vit_max_time_steps=1000,
        vit_dropout=0.1,
        vit_dropout_emb=0.1,
        # --- projection ---
        d_proj=512,
        # --- ignored (consumed by spec2conf_base if used) ---
        **kwargs,
    ):
        super().__init__()

        # --- Node encoder: z → one-hot(in_node_nf) ---
        self.node_encoder = nn.Embedding(100, in_node_nf)

        self.molecular_encoder = EGNN(
            in_node_nf=in_node_nf,
            in_edge_nf=0,
            hidden_nf=hidden_nf,
            n_layers=n_layers,
            coords_weight=coords_weight,
            attention=attention,
            node_attr=1,
            output_size=mol_output_size,
        )

        self.spectral_encoder = ViT(
            patch_size=vit_patch_size,
            num_layers=vit_num_layers,
            h_dim=vit_h_dim,
            num_heads=vit_num_heads,
            output_size=mol_output_size,
            d_ff=vit_d_ff,
            max_time_steps=vit_max_time_steps,
            use_clf_token=True,
            dropout=vit_dropout,
            dropout_emb=vit_dropout_emb,
        )

        self.molecular_proj = nn.Sequential(
            nn.Linear(mol_output_size, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(mol_output_size, d_proj), nn.ReLU(), nn.Linear(d_proj, d_proj),
        )
        self.logit_scale = nn.Parameter(torch.tensor(4.6))

    # ----------------------------------------------------------
    #  Molecular encoder (PyG Data → per-graph features)
    # ----------------------------------------------------------

    def get_molecular_embedding(self, inputs):
        pos = inputs.pos        # (total_atoms, 3)
        z = inputs.x.squeeze()  # (total_atoms,)
        batch = inputs.batch    # (total_atoms,)
        device = z.device

        num_graphs = int(batch.max().item()) + 1

        # pad to uniform n_nodes for EGNN (requires fixed n_nodes per batch)
        counts = scatter(torch.ones_like(batch), batch, reduce="sum")
        max_n = int(counts.max().item())

        # node features: (total_atoms, in_node_nf)
        h0 = self.node_encoder(z)

        # padded coords: (num_graphs, max_n, 3)
        padded_h0 = h0.new_zeros(num_graphs, max_n, h0.size(1))
        padded_pos = pos.new_zeros(num_graphs, max_n, 3)
        padded_mask = h0.new_zeros(num_graphs, max_n, 1)

        # fill — use scatter to avoid Python loops
        idx = torch.arange(z.size(0), device=device)
        mol_idx = batch
        atom_idx_in_mol = torch.zeros_like(batch)
        # compute per-graph running atom index
        offset = torch.zeros(num_graphs, dtype=torch.long, device=device)
        atom_idx_in_mol = torch.empty_like(batch)
        for i in range(num_graphs):
            mask_i = batch == i
            n_i = mask_i.sum().item()
            atom_idx_in_mol[mask_i] = torch.arange(n_i, device=device)
            offset[i] = n_i

        padded_h0[mol_idx, atom_idx_in_mol] = h0
        padded_pos[mol_idx, atom_idx_in_mol] = pos
        padded_mask[mol_idx, atom_idx_in_mol] = 1.0

        # flatten: (num_graphs * max_n, ...)
        h0_flat = padded_h0.view(-1, h0.size(1))
        pos_flat = padded_pos.view(-1, 3)
        mask_flat = padded_mask.view(-1, 1)

        # fully-connected edges within each molecule (with self-loops)
        rows_list, cols_list = [], []
        for i in range(num_graphs):
            base = i * max_n
            idxs = torch.arange(max_n, device=device)
            aa = idxs.unsqueeze(1).expand(max_n, max_n).reshape(-1) + base
            bb = idxs.unsqueeze(0).expand(max_n, max_n).reshape(-1) + base
            rows_list.append(aa)
            cols_list.append(bb)
        rows = torch.cat(rows_list)
        cols = torch.cat(cols_list)
        edges = [rows, cols]
        edge_mask = torch.ones(rows.size(0), 1, device=device)

        mol_features = self.molecular_encoder(
            h0=h0_flat,
            x=pos_flat,
            edges=edges,
            edge_attr=None,
            node_mask=mask_flat,
            edge_mask=edge_mask,
            n_nodes=max_n,
        )
        mol_features = F.normalize(mol_features, p=2, dim=1)
        return mol_features

    # ----------------------------------------------------------
    #  Spectral encoder
    # ----------------------------------------------------------

    def get_spectral_embedding(self, inputs):
        spec = inputs.get("raman", inputs.get("ir"))  # (B, 1, L) or (B, L)
        if spec.dim() == 2:
            spec = spec.unsqueeze(1)
        spec = spec.unsqueeze(1)  # (B, 1, 1, L) — ViT expects 4-D
        features = self.spectral_encoder(spec)
        features = F.normalize(features, p=2, dim=1)
        return features

    # ----------------------------------------------------------
    #  Forward (Vib2Conf interface)
    # ----------------------------------------------------------

    def forward(self, inputs, return_loss=True, return_proj_output=False):
        mol_feat = self.molecular_proj(self.get_molecular_embedding(inputs))
        spec_feat = self.spectral_proj(self.get_spectral_embedding(inputs))

        result = {}
        if return_loss:
            mol_n = F.normalize(mol_feat, p=2, dim=1)
            spec_n = F.normalize(spec_feat, p=2, dim=1)
            sim = mol_n @ spec_n.t() * self.logit_scale.exp()
            # CL loss: symmetric cross-entropy on similarity matrix
            labels = torch.arange(sim.size(0), device=sim.device)
            loss_i = F.cross_entropy(sim, labels)
            loss_t = F.cross_entropy(sim.t(), labels)
            cl_loss = (loss_i + loss_t) / 2
            result["cl_loss"] = cl_loss
            result["loss"] = cl_loss

        if return_proj_output:
            result["molecular_proj_output"] = mol_feat
            result["spectral_proj_output"] = spec_feat

        return result


# ============================================================
#  Registration
# ============================================================

@register_model
def smen(**kwargs):
    return SMEN(**kwargs)
