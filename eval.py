import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from models import build_model
from utils.engine import seed_everything
from utils.dataloader import Dataloader


seed_everything(624)

def compute_recall(topk_indices, num_queries, k_val):
    """
    Compute Recall@k for a given set of top-k indices.
    topk_indices: shape (num_queries, max_k)
    num_queries: total number of queries
    k_val: the K for which to compute recall (e.g., 1, 3, 5)
    """
    if k_val > topk_indices.shape[1]:
        print(f"Warning: k_val ({k_val}) is greater than the number of available ranked items ({topk_indices.shape[1]}). "
              "Adjusting k_val to available items.")
        k_val = topk_indices.shape[1]

    correct = 0
    for i in range(num_queries):
        # Check if the query's true index (i) is within the top k_val predictions for that query
        if i in topk_indices[i, :k_val]:
            correct += 1
    recall_at_k = correct / num_queries
    return recall_at_k


def get_args():
    
    parser = argparse.ArgumentParser(description="Evaluate the model on the test set.")
    parser.add_argument('--ckpt_path', type=str, required=False, help='Path to the checkpoint file.',
                        default='checkpoints/vb_confs/ir/spec2conf_equiformer_moe3/2026-02-24-07-51-019eb6/epoch147.pth')
    return parser.parse_args()

args = get_args()

if __name__ == "__main__":
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ckpt_path = args.ckpt_path

    dataset_name = ckpt_path.split('/')[1]
    modalities = ckpt_path.split('/')[2].split('-')
    model_name = ckpt_path.split('/')[3]

    model = build_model(model_name).to(device)
    ckpt = torch.load(f'./{ckpt_path}', map_location=device, weights_only=True)

    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}

    new_state_dict = {}
    for k, v in ckpt.items():
        # 核心替换逻辑
        new_key = k.replace('multi_encoder_layers', 'layers').replace('multi_enc_norm', 'norm')
        new_state_dict[new_key] = v
        
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)


    dataloader = Dataloader(ds=dataset_name, 
                            data_dir=f'./datasets', 
                            device=device, 
                            force_reload=False)

    test_loader = dataloader.generate_dataloader(mode='test', batch_size=128)

    all_molecular_embeddings = []
    all_spectra_embeddings = []

    model.eval()
    with torch.no_grad():
        
        for batch in tqdm(test_loader):

            batch['raman'] = batch['raman'].to(device) if 'raman' in modalities else None
            batch['ir'] = batch['ir'].to(device) if 'ir' in modalities else None
            
            output = model(inputs=batch.to(device), return_proj_output=True)
            
            all_molecular_embeddings.append(output['molecular_proj_output'].detach().cpu())
            all_spectra_embeddings.append(output['spectral_proj_output'].detach().cpu())
            

    all_molecular_embeddings = torch.cat(all_molecular_embeddings, dim=0)
    all_spectra_embeddings = torch.cat(all_spectra_embeddings, dim=0)

    all_molecular_embeddings = F.normalize(all_molecular_embeddings, p=2, dim=1)
    all_spectra_embeddings = F.normalize(all_spectra_embeddings, p=2, dim=1)


    simi_matrix = torch.mm(all_spectra_embeddings, all_molecular_embeddings.T)

    num_queries = simi_matrix.size(0)
    max_k_eval = 5 # Default for initial retrieval display, or args.topk if it's larger

    _, initial_topk_indices = simi_matrix.topk(max_k_eval, dim=1, largest=True, sorted=True)

    top1_recall_initial = compute_recall(initial_topk_indices, num_queries, k_val=1)
    top3_recall_initial = compute_recall(initial_topk_indices, num_queries, k_val=3)
    top5_recall_initial = compute_recall(initial_topk_indices, num_queries, k_val=5)

    print(f"Retrieval Recall@1: {top1_recall_initial:.4f}")
    print(f"Retrieval Recall@3: {top3_recall_initial:.4f}")
    print(f"Retrieval Recall@5: {top5_recall_initial:.4f}")