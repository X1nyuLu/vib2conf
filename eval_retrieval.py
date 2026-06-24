"""
eval_retrieval.py — Contrastive learning model (Equiformer/TranSpec) retrieval evaluation.

Auto-parses dataset / task / model_name from the checkpoint path.
Supports equiformer key remapping (multi_encoder_layers -> layers).
Outputs Recall@1/3/5, optionally saves a DataFrame with SMILES retrieval results.

Usage:
    python eval_retrieval.py \\
        --ckpt checkpoints/geom_qm9s/raman/spec2conf_equiformer_moe_balance0001/xxx/epoch148.pth \\
        [--ds geom_qm9s] [--model spec2conf_equiformer_moe_balance0001] [--task raman] \\
        [--save results.pickle]
"""

import argparse
import sys
import os
import pickle

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import lmdb
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from utils.engine import seed_everything
from utils.dataloader import Dataloader


def get_args():
    parser = argparse.ArgumentParser('Retrieval Evaluation')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--ds', type=str, default=None,
                        help='Dataset name (auto-extracted from ckpt path if omitted)')
    parser.add_argument('--task', type=str, default=None,
                        help='Task (auto-extracted from ckpt path if omitted)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (auto-extracted from ckpt path if omitted)')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=624)
    parser.add_argument('--save', type=str, default=None,
                        help='Save result DataFrame to .pickle')
    parser.add_argument('--data-dir', type=str, default='datasets')
    parser.add_argument('--topk', type=int, default=5,
                        help='Max K for recall evaluation')
    # key remapping for equiformer checkpoints
    parser.add_argument('--remap', type=str, nargs='*', default=[],
                        help='Key remapping rules: old=new old2=new2 ...  '
                             'e.g. multi_encoder_layers=layers multi_enc_norm=norm')
    return parser.parse_args()


def parse_ckpt_path(ckpt_path):
    """Parse dataset / task / model_name from checkpoint path.
    Expected format: .../checkpoints/<ds>/<task>/<model>/<run>/epoch*.pth
    """
    parts = ckpt_path.replace('\\', '/').split('/')
    try:
        # search backward for the "checkpoints" segment
        ckpt_idx = next(i for i, p in enumerate(parts) if 'checkpoint' in p)
        rel = parts[ckpt_idx + 1:]  # path segments after "checkpoints"
        ds, task, model = rel[0], rel[1], rel[2]
        return ds, task, model
    except (StopIteration, IndexError):
        return None, None, None


def load_model(ckpt_path, model_name, device, remap_rules):
    model = build_model(model_name).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}

    if remap_rules:
        remap = {}
        for rule in remap_rules:
            if '=' in rule:
                old, new = rule.split('=', 1)
                remap[old] = new
        new_state_dict = {}
        for k, v in ckpt.items():
            nk = k
            for old, new in remap.items():
                nk = nk.replace(old, new)
            new_state_dict[nk] = v
        ckpt = new_state_dict

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f'  Missing keys: {missing}')
    if unexpected:
        print(f'  Unexpected keys: {unexpected}')
    model.eval()
    return model


def compute_recall(topk_indices, num_queries, k_val):
    """Compute Recall@k."""
    if k_val > topk_indices.shape[1]:
        k_val = topk_indices.shape[1]
    correct = 0
    for i in range(num_queries):
        if i in topk_indices[i, :k_val]:
            correct += 1
    return correct / num_queries


@torch.no_grad()
def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    # ---- Parse arguments ----
    ckpt_ds, ckpt_task, ckpt_model = parse_ckpt_path(args.ckpt)
    ds = args.ds or ckpt_ds
    task = args.task or ckpt_task
    model_name = args.model or ckpt_model

    if not all([ds, task, model_name]):
        print('Error: cannot parse dataset/task/model from checkpoint path. '
              'Provide --ds, --task, --model explicitly.')
        sys.exit(1)

    print(f'Dataset : {ds}')
    print(f'Task    : {task}')
    print(f'Model   : {model_name}')
    print(f'Checkpoint: {args.ckpt}')

    modalities = task.split('-')

    # ---- Load model ----
    print('\nLoading model ...')
    model = load_model(args.ckpt, model_name, device, args.remap)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model loaded. Params: {total_params / 1e6:.2f}M')

    # ---- Load test set ----
    print(f'\nLoading test set ...')
    dataloader = Dataloader(
        ds=ds,
        data_dir=args.data_dir,
        target_keys=modalities,
        device=device,
        force_reload=False,
    )
    test_loader = dataloader.generate_dataloader(
        mode='test', batch_size=args.batch_size
    )

    # ---- Inference ----
    print('Running inference ...')
    all_mol_emb = []
    all_spec_emb = []
    all_ir_emb = []
    all_raman_emb = []

    for batch in tqdm(test_loader):
        batch['raman'] = batch['raman'].to(device) if 'raman' in modalities else None
        batch['ir'] = batch['ir'].to(device) if 'ir' in modalities else None

        output = model(inputs=batch.to(device), return_proj_output=True)

        all_mol_emb.append(output['molecular_proj_output'].detach().cpu())
        all_spec_emb.append(output['spectral_proj_output'].detach().cpu())

        # Collect per-modality embeddings for dual retrieval
        if 'ir_proj_output' in output:
            all_ir_emb.append(output['ir_proj_output'].detach().cpu())
        if 'raman_proj_output' in output:
            all_raman_emb.append(output['raman_proj_output'].detach().cpu())

    all_mol_emb = torch.cat(all_mol_emb, dim=0)
    all_spec_emb = torch.cat(all_spec_emb, dim=0)

    all_mol_emb = F.normalize(all_mol_emb, p=2, dim=1)
    all_spec_emb = F.normalize(all_spec_emb, p=2, dim=1)

    N = all_mol_emb.size(0)
    is_dual = len(all_ir_emb) > 0 and len(all_raman_emb) > 0
    print(f'Inference done: {N} samples, emb_dim={all_mol_emb.size(1)}, dual={is_dual}')

    if is_dual:
        all_ir_emb = F.normalize(torch.cat(all_ir_emb, dim=0), p=2, dim=1)
        all_raman_emb = F.normalize(torch.cat(all_raman_emb, dim=0), p=2, dim=1)

        # Geometric mean retrieval: cbrt(sim_g_ir * sim_g_raman * sim_ir_raman)
        sim_g_ir = torch.mm(all_mol_emb, all_ir_emb.T)
        sim_g_raman = torch.mm(all_mol_emb, all_raman_emb.T)
        sim_ir_raman = torch.mm(all_ir_emb, all_raman_emb.T)

        simi_s2m = np.cbrt(sim_g_ir * sim_g_raman * sim_ir_raman)
    else:
        # Single-modality: standard cosine similarity
        simi_s2m = torch.mm(all_spec_emb, all_mol_emb.T)

    max_k = min(args.topk, N)
    _, topk_s2m = simi_s2m.topk(max_k, dim=1)

    print('\n========== Retrieval Recall (Spectrum → Molecule) ==========')
    recalls_s2m = {}
    for k in [1, 3, 5]:
        if k > N:
            continue
        r = compute_recall(topk_s2m, N, k)
        recalls_s2m[k] = r
        print(f'  Recall@{k}: {r:.4f}')

    # ---- Retrieval: molecule → spectrum ----
    _, topk_m2s = simi_s2m.T.topk(max_k, dim=1)

    print('\n========== Retrieval Recall (Molecule → Spectrum) ==========')
    recalls_m2s = {}
    for k in [1, 3, 5]:
        if k > N:
            continue
        r = compute_recall(topk_m2s, N, k)
        recalls_m2s[k] = r
        print(f'  Recall@{k}: {r:.4f}')

    # ---- Optionally load raw LMDB data for SMILES-level analysis and save ----
    if args.save:
        print(f'\nLoading LMDB raw data for SMILES analysis ...')
        data_dir_full = os.path.join(args.data_dir, ds)
        lmdb_path = os.path.join(data_dir_full, 'test.lmdb')
        if os.path.exists(lmdb_path):
            db = lmdb.open(lmdb_path, subdir=False, lock=False, map_size=int(1e11))
            with db.begin() as txn:
                raw_data = list(txn.cursor())
            db.close()
            df = pd.DataFrame([pickle.loads(item[1]) for item in raw_data])

            # add top-k indices and SMILES hit analysis
            df['predict_conf_idx'] = topk_s2m.numpy().tolist()

            all_smiles = df['smiles'].values
            topk_smiles = all_smiles[topk_s2m.numpy()]
            df['predict_smiles'] = topk_smiles.tolist()

            targets = df['smiles'].values.reshape(-1, 1)
            for k in range(1, max_k + 1):
                df[f'top{k}_smiles_hit'] = np.any(
                    topk_smiles[:, :k] == targets, axis=1
                )

            print(f'  Top-1 SMILES accuracy: {df["top1_smiles_hit"].mean():.4f}')

            df.to_pickle(args.save)
            print(f'  DataFrame saved to {args.save}')
        else:
            print(f'  Warning: {lmdb_path} not found, skipping SMILES analysis.')


if __name__ == '__main__':
    main()
