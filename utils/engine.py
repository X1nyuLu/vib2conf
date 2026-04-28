'''
basic functions
'''


import os
import random
import logging

from tqdm import tqdm
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Monitor:
    def __init__(self, mode='max', delta=0.0, keep_n=3):

        self.mode = mode
        self.best_score = None
        self.delta = delta
        self.val_score = None
        self.keep_n = keep_n
        self.recent_checkpoints = []
        
    def __call__(self, epoch_score, model, model_path):
        should_save = False
        
        if self.mode == 'min':
            score = -1.0 * epoch_score
        else:
            score = float(epoch_score)

        if self.best_score is None:
            self.best_score = score
            should_save = True

        elif score > self.best_score + self.delta:
            self.best_score = score
            should_save = True

        if should_save:
            self._save_and_cleanup(epoch_score, model, model_path)
            
    def _save_and_cleanup(self, epoch_score, model, model_path):
        torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score
        self.recent_checkpoints.append(model_path)

        if len(self.recent_checkpoints) > 0 and len(self.recent_checkpoints) > self.keep_n:
            oldest_path = self.recent_checkpoints.pop(0)
            if os.path.exists(oldest_path):
                os.remove(oldest_path)                        
                        
    def _display_best_score(self):
        if self.mode == 'min':
            return -1.0 * self.best_score
        return self.best_score
                
class BaseEngine:
    def __init__(self, train_loader=None, eval_loader=None, test_loader=None,
                 optimizer=None, scheduler=None,
                 model=None, model_ema=None, device='cpu', device_rank=0, ddp=False, task=None, **kwargs):

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.model = model
        self.model_ema = model_ema
        self.device = device
        self.device_rank = device_rank
        self.ddp = ddp
        self.task = task

    def train_epoch(self, epoch, max_grad_norm=1.0):
        train_losses = AverageMeter()
        self.model.train()

        if self.scheduler:
            self.scheduler.step(epoch)
            
        bar = tqdm(self.train_loader) if self.device_rank == 0 else self.train_loader
        for batch in bar:
            self.optimizer.zero_grad()
            
            batch['raman'] = batch['raman'].to(self.device) if 'raman' in self.task else None
            batch['ir'] = batch['ir'].to(self.device) if 'ir' in self.task else None
            batch['hnmr'] = batch['hnmr'].to(self.device) if 'hnmr' in self.task else None
            batch['cnmr'] = batch['cnmr'].to(self.device) if 'cnmr' in self.task else None
            
            output = self.model(inputs=batch.to(self.device))
                
            loss = output['loss']
            loss.backward()
            
            # Gradient Clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
            
            self.optimizer.step()
            if self.model_ema is not None:
                self.model_ema.update(self.model)
            
            train_losses.update(loss.detach().item(), batch.batch.max().item()+1)
            if self.device_rank == 0:
                bar.set_description(
                    f'Epoch{epoch:4d}, train loss:{train_losses.avg:6f}')

        if self.device_rank == 0:
            logging.info(f'Epoch{epoch:4d}, train loss:{train_losses.avg:6f}')
        return train_losses.avg

    @torch.no_grad()
    def eval_epoch(self, epoch):

        eval_losses = AverageMeter()
        eval_losses_cl = AverageMeter()
        eval_losses_match = AverageMeter()
        eval_acc = AverageMeter()

        self.model.eval()
        
        all_molecular_embeddings = []
        all_spectra_embeddings = []

        bar = tqdm(self.eval_loader) if self.device_rank == 0 else self.eval_loader

        for batch in bar:
            
            self.optimizer.zero_grad()
            
            batch['raman'] = batch['raman'].to(self.device) if 'raman' in self.task else None
            batch['ir'] = batch['ir'].to(self.device) if 'ir' in self.task else None
            batch['hnmr'] = batch['hnmr'].to(self.device) if 'hnmr' in self.task else None
            batch['cnmr'] = batch['cnmr'].to(self.device) if 'cnmr' in self.task else None
            
            if self.model_ema is not None:
                output = self.model_ema.module(inputs=batch.to(self.device), return_proj_output=True)
            else:
                output = self.model(inputs=batch.to(self.device), return_proj_output=True)
                
            eval_losses.update(output['loss'].item(), batch.batch.max().item()+1)
            eval_losses_cl.update(output['cl_loss'].item(), batch.batch.max().item()+1)
            
            all_molecular_embeddings.append(output['molecular_proj_output'].detach().cpu())
            all_spectra_embeddings.append(output['spectral_proj_output'].detach().cpu())
            
            if 'matching_loss' in output:
                eval_losses_match.update(output['matching_loss'].item(), batch.batch.max().item()+1)
                matching_accuracy = output['matching_accuracy'].cpu().item()
                eval_acc.update(matching_accuracy, batch.batch.max().item()+1)
            
            if self.device_rank == 0:
                val_acc_desc = f", valid acc:{eval_acc.avg:6f}" if 'matching_accuracy' in output else ""
                bar.set_description(
                    f'Epoch{epoch:4d}, valid loss:{eval_losses.avg:6f}{val_acc_desc}')
                    
        all_molecular_embeddings = torch.cat(all_molecular_embeddings, dim=0)
        all_spectra_embeddings = torch.cat(all_spectra_embeddings, dim=0)
        
        all_molecular_embeddings = torch.nn.functional.normalize(all_molecular_embeddings, p=2, dim=1)
        all_spectra_embeddings = torch.nn.functional.normalize(all_spectra_embeddings, p=2, dim=1)

        simi_matrix = torch.mm(all_molecular_embeddings, all_spectra_embeddings.T)
        molecule_to_spectrum_recall = compute_recall(simi_matrix, k=1)
        spectrum_to_molecule_recall = compute_recall(simi_matrix.T, k=1)
        if self.device_rank == 0:
            val_acc_desc = f", valid acc:{eval_acc.avg:6f}" if 'matching_accuracy' in output else ""
            logging.info(
                f'Epoch{epoch:4d}, eval loss:{eval_losses.avg:6f}, molecule_to_spectrum_recall:{molecule_to_spectrum_recall:6f}, spectrum_to_molecule_recall:{spectrum_to_molecule_recall:6f}{val_acc_desc}')
    
        return {'loss':eval_losses.avg, 
                'cl_loss':eval_losses_cl.avg, 
                'recall':spectrum_to_molecule_recall, 
                'matching_loss':eval_losses_match.avg if "matching_loss" in output else None, 
                'acc':eval_acc.avg if "matching_accuracy" in output else None}
    
    
def compute_recall(similarity_matrix, k, verbose=False):
    num_queries = similarity_matrix.size(0)
    _, topk_indices = similarity_matrix.topk(k, dim=1, largest=True, sorted=True)
    correct = 0
    for i in range(num_queries):
        if i in topk_indices[i]:
            correct += 1
    recall_at_k = correct / num_queries
    
    if verbose:
        print(f'recall@{k}:{recall_at_k:.5f}')
    else:
        return recall_at_k
    

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True