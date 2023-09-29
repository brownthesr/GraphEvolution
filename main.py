from lightning.pytorch import loggers as pl_loggers
from models import GraphAutoEncoder
from utils import generate_ssbm,draw_graph, generate_dcbm
from datasets import SBMGraphDataset,GraphTDataset
import lightning.pytorch as pl
from Graph_transformer.data import GraphDataset
from torch_geometric.loader import DataLoader

# graph, communities = generate_ssbm(9,3,.7,.1)
graph, communities,_ = generate_dcbm(30,3,.7,.1,5,1.3)
draw_graph(graph,communities,-1)
dataset = GraphDataset(GraphTDataset(30,3,32000*4))
dataloader = DataLoader(dataset=dataset,batch_size=8,num_workers=20)
# from torch.utils.data import DataLoader
# dataset = SBMGraphDataset(30,3,32000*4)
# dataloader = DataLoader(dataset=dataset,batch_size=16,num_workers=20)

lr = 5e-5
num_epochs = 800
eigen_positions = 30
model = GraphAutoEncoder(1024,3,dataset,num_layers=4,model="GCN",
                          lr = lr,use_scheduler=True,num_epochs=num_epochs)

wandb_logger = pl_loggers.WandbLogger(project="Equation_free", name= f"{lr} lr, SAGE, 30 nodes, more eigen, 800, SGD, 64 batch",offline=True)
trainer = pl.Trainer(max_epochs=num_epochs,sync_batchnorm=True,logger = wandb_logger,devices=8, num_nodes=1,strategy="ddp")
trainer.fit(model,train_dataloaders=dataloader)
