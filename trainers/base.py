import logging
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from timm.scheduler import create_scheduler_v2
from timm.utils import ModelEmaV2

from utils.engine import AverageMeter, Monitor, BaseEngine
from utils.dataloader import Dataloader
from trainers import register_function

class BaseTrainer:
    def __init__(self,
                 model, model_save_path=None, device='cpu', ddp=False, rank=-1, config=None,
                 ds=None, task=None, data_dir=None, force_reload=False, use_ema=False,
                 **kwargs):
        
        self.model = model
        self.model_save_path = model_save_path
        self.device = device
        self.ddp = ddp
        self.rank = rank
        self.config = config
        self.ds = ds
        self.task = task
        self.data_dir = data_dir
        self.force_reload = force_reload
        self.use_ema = use_ema
        
    def init_dataset(self):
        if self.rank == 0:
            self.writer = SummaryWriter(self.model_save_path.replace('checkpoints', 'runs'))

        target_keys = self.task.split('-')
        dataloader = Dataloader(
            ds=self.ds, 
            data_dir=self.data_dir,
            target_keys=target_keys, 
            device=self.device,
            force_reload=self.force_reload)
        
        if self.ddp:
            self.train_loader = dataloader.generate_dataloader(
                mode='train',
                batch_size=self.config['batch_size'],
                num_workers=4, 
                ddp=self.ddp)
        else:
            self.train_loader = dataloader.generate_dataloader(
                mode='train',
                batch_size=self.config['batch_size'], 
                num_workers=4)
            
        self.eval_loader = dataloader.generate_dataloader(mode='eval', batch_size=64)
        
    def init_engine(self, Engine, **kwargs):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.config['lr']), weight_decay=float(self.config['weight_decay']))
        scheduler, _ = create_scheduler_v2(
            optimizer, 
            sched=self.config['lr_sched'], 
            num_epochs=self.config['epoch'], 
            warmup_epochs=self.config['warmup_epochs'],
            min_lr=float(self.config['min_lr']),
            warmup_lr=float(self.config['warmup_lr'])
            )

        self.monitor = Monitor(mode='max')
        
        self.engine = Engine(
            train_loader=self.train_loader, 
            eval_loader=self.eval_loader, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            model=self.model, 
            model_ema=ModelEmaV2(self.model, decay=0.999) if self.use_ema else None,
            device=self.device, 
            device_rank=self.rank, 
            ddp=self.ddp, 
            task=self.task,
            **kwargs
            )
        
    def train(self):
        '''
        rewrite this method to train model
        '''
        pass


def train_model(Trainer, Engine, 
                model=None, ds=None, task=None, data_dir=None,
                model_save_path=None, device='cpu', ddp=False, rank=-1, config=None, use_ema=False, **kwargs):
    
    trainer = Trainer(
        model=model,
        model_save_path=model_save_path,
        device=device,
        ddp=ddp,
        rank=rank,
        config=config,
        ds=ds,
        task=task,
        data_dir=data_dir,
        use_ema=use_ema,
        **kwargs,
    )
    
    trainer.init_dataset()
    trainer.init_engine(Engine, **kwargs)
    trainer.train()
    
    
class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def train(self):
        for epoch in range(self.config['epoch']):
            if hasattr(self.train_loader, 'sampler') and isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            if hasattr(self.eval_loader, 'sampler') and isinstance(self.eval_loader.sampler, DistributedSampler):
                self.eval_loader.sampler.set_epoch(epoch)
                
            train_loss = self.engine.train_epoch(epoch)
            eval_output = self.engine.eval_epoch(epoch)

            if self.engine.device_rank == 0:
                self.writer.add_scalar('train_loss', train_loss, epoch)
                self.writer.add_scalar('eval_loss', eval_output['loss'], epoch)
                self.writer.add_scalar('eval_recall', eval_output['recall'], epoch)
                if eval_output['acc'] is not None:
                    self.writer.add_scalar('eval_acc', eval_output['acc'], epoch)
                
                self.monitor(
                    epoch_score=eval_output['recall'], 
                    model=self.engine.model_ema.module if self.use_ema else self.model, 
                    model_path=f"{self.model_save_path}/epoch{epoch}.pth"
                    )
                
                val_acc_desc = f", valid acc:{eval_output['acc']:6f}" if eval_output['acc'] is not None else ""
                logging.info(
                    f"Epoch{epoch:4d}, valid loss:{eval_output['loss']:6f}, valid recall:{eval_output['recall']:6f}{val_acc_desc}, best recall:{self.monitor._display_best_score():6f}")
            
        if self.rank == 0:
            torch.save(self.model.state_dict(), f'{self.model_save_path}/epoch{epoch}.pth')
            print(self.monitor._display_best_score())
        if self.rank == 0:
            self.writer.close()
            
        
@register_function('base')
def train_matching_model(*args, **kwargs):
    return train_model(Trainer, BaseEngine, *args, **kwargs)
