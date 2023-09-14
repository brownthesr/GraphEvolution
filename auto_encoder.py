from lightning.pytorch import loggers as pl_loggers
from models import GraphAutoEncoder
from utils import generate_ssbm,draw_graph, generate_dcbm
from datasets import GraphDataset
import lightning.pytorch as pl
from torch.utils.data import DataLoader

# graph, communities = generate_ssbm(9,3,.7,.1)
graph, communities = generate_dcbm(30,3,.7,.1,5,1.3)
draw_graph(graph,communities,-1)
dataset = GraphDataset(30,3,32000*4)

dataloader = DataLoader(dataset=dataset,batch_size=64,num_workers=20)
model = GraphAutoEncoder(1024,3,dataset,num_layers=4)

wandb_logger = pl_loggers.WandbLogger(project="Equation_free", name= " 4e-6 lr, SAGE, 30 nodes, smaller batch, batch norm",offline=True)
trainer = pl.Trainer(max_epochs=600,sync_batchnorm=True,logger = wandb_logger)
trainer.fit(model,train_dataloaders=dataloader)
