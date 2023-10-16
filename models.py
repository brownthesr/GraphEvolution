import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer as Layer
from torch.nn import TransformerEncoder as Encoder
from torch.nn import Transformer
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, global_add_pool
import lightning as L
from torch_geometric.data import Data
# from Graph_transformer.models import GraphTransformer
from utils import positional_embedding, draw_graph
from data.transforms import batch_to_sequence
import torch.nn.functional as F
NEW_NODE = torch.Tensor([0]).long()
EOS_TOKEN = torch.Tensor([1]).long()
SOS_TOKEN = torch.Tensor([2]).long()


class GNN(nn.Module):   
    def __init__(self,d_in, d_hidden, d_out,n_layers = 2,kind="SAGE", use_batch_norm = False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.use_batch_norm= use_batch_norm
        if use_batch_norm:
            self.batch_norms = nn.ModuleList()
            self.batch_norms.append(nn.norm.GraphNorm(d_hidden) )
        if kind == "SAGE":
            layer = SAGEConv
        elif kind == "GCN":
            layer = GCNConv
        elif kind == "GAT":
            layer = GATConv
        else:
            raise TypeError(f"{kind} is not a valid layer to input")
        self.convs.append(layer(d_in,d_hidden))
        for i in range(n_layers-2):
            self.convs.append(layer(d_hidden,d_hidden))
            if use_batch_norm:
                self.batch_norms.append(nn.norm.GraphNorm(d_hidden) )
        self.convs.append(layer(d_hidden,d_out))

        self.activation = nn.GELU()


        
    def forward(self, databatch):
        """
        Forward pass

        Parameters
        ----------
        A: torch tensor of shape (batch_size, num_nodes, num_nodes)
            The adjacency matrices
        x: torch tensor of shape (batch_size, num_nodes, feature_dim)
            The features of our nodes
        
        Returns
        -------
        torch.Tensor: The forward pass of our model.

        """
        # Set our initial data to our input
        x = databatch.x
        edge_index = databatch.edge_index
        batch = databatch.batch
        data = x

        # Run everything through the graph convolutions and activations
        for i in range(len(self.convs)):
            data = self.convs[i](data,edge_index)
            if self.use_batch_norm:
                data = self.batch_norms[i](data)
            data = self.activation(data)
        
        # pool with respect to batches
        return global_add_pool(data,batch)

class GraphDecoder(nn.Module):
    def __init__(self, d_latent,d_model,n_layers = 5, n_discrete_tokens = 43, n_continuous_dim=2):
        super().__init__()
        # transformer layers
        encoder_layer = Layer(d_model=d_model,nhead=4,batch_first=True)
        self.decoder = Encoder(encoder_layer=encoder_layer,num_layers = n_layers,enable_nested_tensor=True)
        self.d_model = d_model


        self.discrete_embedder = nn.Embedding(n_discrete_tokens,d_model)
        # you would change the next line to be a nn.Linear if we were dealing with continuous states
        self.continuous_embedder = nn.Linear(2,d_model)
        # This converts our latent space to the dimension of our model
        self.latent_embedder = nn.Linear(d_latent,d_model)
        self.device = "cuda"

        self.cont_decode = nn.Linear(d_model,2)
        self.discrete_decode = nn.Linear(d_model,n_discrete_tokens)
    
    def forward(self,input_seq, padding_mask):
        position_emb = positional_embedding(input_seq.shape[-2],self.d_model)
        # print(input_seq.shape)
        # print(padding_mask.shape)
        inf_mask = torch.where(~padding_mask, torch.tensor(float('-inf')), torch.tensor(0.0))

        output = self.decoder(input_seq+ position_emb.to(self.device),Transformer.generate_square_subsequent_mask(input_seq.shape[-2]).to(self.device),
                              src_key_padding_mask=inf_mask,is_causal=True)
        return output

        

class GraphAutoEncoder(L.LightningModule):
    def __init__(self, d_model,d_data,dataset,num_layers, 
                 lr, use_scheduler=False, num_epochs=100,
                 eigen_positions = 8, model = "GCN",n_discrete_tokens=43,n_continuous_dim=21,
                 encoder_hidden_dim = 128, latent_dim = 256,decoder_hidden_dim = 128, n_encoder_layers = 5,
                 n_decoder_layers = 5):
        super().__init__()
        self.lr = lr
        self.d_data = d_data
        self.num_epochs = num_epochs
        self.use_scheduler = use_scheduler

        # # is a NN to decode the features
        # self.decode_data = nn.Sequential(
        #                                 nn.Linear(d_model,d_model),
        #                                 nn.ReLU(),
        #                                 nn.Linear(d_model,d_data)
        #                                 )
        
        # # This one is to decode the edges
        # self.decode_edge = nn.Sequential(
        #                                 nn.Linear(d_model,d_model),
        #                                  nn.ReLU(),
        #                                  nn.Linear(d_model,5)
        #                                  )
        
        self.embedding_size = d_model

        self.encoder = GNN(d_data,encoder_hidden_dim,latent_dim,n_layers=n_encoder_layers)
        self.decoder = GraphDecoder(latent_dim,decoder_hidden_dim,n_decoder_layers,n_continuous_dim=n_continuous_dim,n_discrete_tokens=n_discrete_tokens)

        # define two different losses
        self.continous_criterion = nn.CrossEntropyLoss()# Target for our features
        self.discrete_criterion = nn.CrossEntropyLoss()# Target for our graph structure

        # debugging
        self.dataset = dataset

    def forward(self,sequence):
        raise NotImplementedError("I should not be called, use the encoder and decoder directly")
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(lr=self.lr,params = self.parameters())
        if self.use_scheduler:
            # sch  = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr,div_factor=20, steps_per_epoch=2000, epochs=80, final_div_factor=10)
            sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = self.trainer.num_training_batches, )
            lr_scheduler_config = {
                # REQUIRED: The scheduler instance
                "scheduler": sch,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "val_loss",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            }
            return ([opt], [lr_scheduler_config])
        else:
            return opt
    
    def training_step(self, batch, batch_idx):
        # For each item in the batch, you can modify it to contain more spectral coordinates with
        # data.transforms.spectral_transform(data,k)
        total_d_loss = 0
        total_c_loss = 0
        for time_step, mini_batch in enumerate(batch):
            # in here we process each time step and save the latent vectors
            latent_vectors = self.encoder(mini_batch)

            # First we get our sequence representations of the graph (Batch, Max_sequence_length, embedding_dim)
            graph_sequences, padding_mask, discrete_mask, continuous_mask, cont_target, discrete_target = batch_to_sequence(mini_batch, self.decoder.discrete_embedder,
                                                                                                                            self.decoder.continuous_embedder)
            # We remove the end of sequence token as we do not want to predict that
            input_sequences = graph_sequences[:,:-1]
            # we replace the start of sequence token with the latent embedding
            input_sequences[torch.arange(len(input_sequences)),0] = self.decoder.latent_embedder(latent_vectors)
            
            decoded_output = self.decoder(input_sequences,padding_mask[:,:-1].bool())

            cont_output = self.decoder.cont_decode(decoded_output[continuous_mask[:,1:]])
            discrete_output = self.decoder.discrete_decode(decoded_output[discrete_mask[:,1:]])
            discrete_loss = self.discrete_criterion(discrete_output,discrete_target)
            continuous_loss = self.continous_criterion(cont_output,cont_target.argmax(dim=-1))
            total_d_loss += discrete_loss
            total_c_loss += continuous_loss
        self.log("Graph loss", total_d_loss)
        self.log("Node attribute loss", total_c_loss)
        self.log("Lr",self.trainer.optimizers[0].param_groups[0]['lr'])
        return total_c_loss+total_d_loss

    def on_train_epoch_end(self):
        """
        Samples from our model at the start of every epoch.
        """
        return
        with torch.no_grad():
            print("Sampling")
            data = self.dataset[0]
            A = data.adj
            A = A.reshape(30,30)
            x = data.x
            x = x.reshape(30,self.d_data)
            draw_graph(A.numpy(),torch.argmax(x,dim=-1).cpu().detach().numpy(),f"{self.current_epoch}")
            A = A.unsqueeze(0).to(self.device)
            x = x.unsqueeze(0).to(self.device)
            latent_representation = self.encoder(A,torch.argmax(x,dim=-1))
            
            

            sequence  = latent_representation.unsqueeze(1)#self.embedder(SOS_TOKEN.to(self.device)).unsqueeze(0).to(self.device)
            new_node_emb = self.embedder(NEW_NODE.to(self.device)).unsqueeze(0).to(self.device)
            new_edge = self.embedder(NEW_EDGE.to(self.device)).squeeze(0)
            no_edge = self.embedder(NO_EDGE.to(self.device)).squeeze(0)
            x = []
            A = torch.eye(30)
            for i in range(30):
                sequence = torch.hstack((sequence,new_node_emb))
                output = self(sequence)
                latent_representation = self.decode_data(output[0,-1,:]).round().float()
                x.append(latent_representation)
                # print(sequence.shape,self.expander(latent_representation).unsqueeze(0).unsqueeze(0).shape)
                sequence = torch.hstack((sequence,self.expander(latent_representation).unsqueeze(0).unsqueeze(0)))

                for j in range(i):
                    output = self(sequence)
                    edge = self.decode_edge(output[0,-1,:])
                    edge_probs =F.softmax(edge[:2],dim=-1)
                    choice = torch.multinomial(edge_probs,1)
                    
                    if choice == 1:
                        sequence = torch.hstack((sequence,new_edge.unsqueeze(0).unsqueeze(0)))
                        A[i,j] = 1
                    else:
                        sequence = torch.hstack((sequence,no_edge.unsqueeze(0).unsqueeze(0)))
                        A[i,j] = 0
            A = A + A.T
            x = torch.vstack(x)
            graph = A.numpy()
            # print(x)
            communities = torch.argmax(x,dim=-1).cpu().detach().numpy()
            draw_graph(graph,communities,f"{self.current_epoch}_reconstructed")
