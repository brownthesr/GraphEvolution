from typing import Any
import torch
import numpy as np
from torch.nn import Embedding
import torch.nn as nn
from tqdm import tqdm
from itertools import permutations
import math 
import networkx as nx
import torch.nn.functional as F
from scipy.stats import poisson
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.nn import TransformerEncoderLayer as Layer
from torch.nn import TransformerEncoder as Encoder
from torch.nn import Transformer
from lightning.pytorch import loggers as pl_loggers
from models import GraphEncoder,GraphAutoEncoder
from utils import generate_ssbm,draw_graph
from datasets import GraphDataset

import lightning as L
import matplotlib.pyplot as plt
import lightning.pytorch as pl


graph, communities = generate_ssbm(9,3,.7,.1)
# graph, communities = generate_dcbm(30,3,.7,.1,5,1.3)
draw_graph(graph,communities,-1)
dataset = GraphDataset(9,3,32000*4)

dataloader = DataLoader(dataset=dataset,batch_size=16,num_workers=3)
model = GraphAutoEncoder(1024,3,dataset,num_layers=16)

wandb_logger = pl_loggers.WandbLogger(project="Equation_free", name= "test",offline=True)
trainer = pl.Trainer(max_epochs=200,sync_batchnorm=True,logger = wandb_logger)
trainer.fit(model,train_dataloaders=dataloader)
