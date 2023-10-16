import torch
from data.dataset import SISDataset
from torch_geometric.loader import DataLoader
from models import GNN,GraphDecoder,GraphAutoEncoder
from lightning.pytorch import loggers as pl_loggers
import lightning.pytorch as pl

dataset = SISDataset("./data/small_dataset/sis_sequences_2.pt")
dataloader = DataLoader(dataset,batch_size=4,drop_last=True)

num_epochs = 800
model = GraphAutoEncoder(256,21,dataset,2,4e-4,num_epochs=num_epochs,use_scheduler=True)
wandb_logger = pl_loggers.WandbLogger(project="Equation_free", name= f"large Dataset SIS, longer, 4e-4 lr",offline=True)

trainer = pl.Trainer(max_epochs=num_epochs,sync_batchnorm=True,logger = wandb_logger,devices=4, num_nodes=1,strategy="ddp")
trainer.fit(model,train_dataloaders=dataloader)