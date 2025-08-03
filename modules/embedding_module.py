import torch
from torch import nn
import numpy as np
import math

from model.temporal_attention import TemporalAttentionLayer

# Embedding module takes in nn - assuming that is neural network
# nn.Module is the basic building block for ALL neural networks in pytorch
# Not sure if this embedding module is implemented correctly, could be used for the ablation, 
# showing when memory is turned off
class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.edge_features = edge_features
    # self.memory = memory
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return NotImplemented

# Identity embedding uses memory directly as node embedding (as described in the paper)
class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return memory[source_nodes, :]

# Time projection embedding where W are the learnable parameters 
class TimeEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
    # now using the methods from the super class, allowing the use of the parameters, 
    # and maybe overwritting the Not Implemented compute_embedding
    super(TimeEmbedding, self).__init__(node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features, embedding_dimension, device, dropout)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None, use_time_proj=True):
    
    # As seen in the paper this is the temporal projection embedding given as memory * (1 + delta time * trainable params W)
    # Method used by JODIE paper 
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))
    return source_embeddings


# class GraphEmbedding(EmbeddingModule):
#   def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
#                n_node_features, n_edge_features, n_time_features, embedding_dimension, device, n_heads=2, dropout=0.1, use_memory=True):
   
#     super(GraphEmbedding, self).__init__(node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
#                                          n_node_features, n_edge_features, n_time_features, embedding_dimension, device, dropout)
    
#     self.use_memory = use_memory
#     self.device = device

#   # Main Compute Embedding funtion that is overwritten depending on the chosen embedding type 
#   def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
#                         use_time_proj=True):
#     """Recursive implementation of curr_layers temporal graph attention layers.

#     src_idx_l [batch_size]: users / items input ids.
#     cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
#     curr_layers [scalar]: number of temporal convolutional layers to stack.
#     num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
#     """

#     assert (n_layers >= 0)

#     source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
#     timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

#     # query node always has the start time -> time span == 0
#     source_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))

#     source_node_features = self.node_features[source_nodes_torch, :]
#     # print(source_node_features.shape)
#     # print(memory[source_nodes, :].shape)
#     print("source node features", source_node_features.shape)
#     if self.use_memory:
#       source_node_features = memory[source_nodes, :] + source_node_features
#     print("Layer Level: ", n_layers)
#     if n_layers == 0:
#       return source_node_features
#     else:
#       # Computing the embedding of the source node
#       print("Computing Souce Node Convolved Embeddings")
#       source_node_conv_embeddings = self.compute_embedding(memory, source_nodes,
#                                                            timestamps, n_layers=n_layers - 1,
#                                                            n_neighbors=n_neighbors)

#       print("Fetching Temporal Neighbours")
#       neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor( source_nodes, timestamps, n_neighbors=n_neighbors)
#       print("Neighbours retrieved, shape: ", neighbors.shape)

#       neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

#       edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

#       edge_deltas = timestamps[:, np.newaxis] - edge_times

#       edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

#       neighbors = neighbors.flatten()
      
#       # Computing the embedding of the neighbour nodes
#       neighbor_embeddings = self.compute_embedding(memory,
#                                                    neighbors,
#                                                    np.repeat(timestamps, n_neighbors),
#                                                    n_layers=n_layers - 1,
#                                                    n_neighbors=n_neighbors)

#       effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
#       neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
#       edge_time_embeddings = self.time_encoder(edge_deltas_torch)

#       edge_features = self.edge_features[edge_idxs, :]

#       mask = neighbors_torch == 0

#       source_embedding = self.aggregate(n_layers, source_node_conv_embeddings,
#                                         source_nodes_time_embedding,
#                                         neighbor_embeddings,
#                                         edge_time_embeddings,
#                                         edge_features,
#                                         mask)

#       return source_embedding

#   def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
#                 neighbor_embeddings,
#                 edge_time_embeddings, edge_features, mask):
#     return NotImplemented

class GraphEmbedding(EmbeddingModule):
    def __init__(self,
                 node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features,
                 embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):

        super().__init__(node_features, edge_features, memory, neighbor_finder, time_encoder,
                         n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device, dropout)

        # 1 Shared-width projections
        self.node_proj = nn.Linear(n_node_features, embedding_dimension, bias=False)
        self.edge_proj = nn.Linear(n_edge_features, embedding_dimension, bias=False)
        self.time_proj = nn.Linear(n_time_features, embedding_dimension, bias=False)

        # Memory projection (optional)
        self.use_memory = use_memory
        if self.use_memory:
            mem_width = memory.memory.shape[1]        # <- safe tensor shape
            self.memory_proj = (nn.Identity() if mem_width == embedding_dimension
                                else nn.Linear(mem_width, embedding_dimension, bias=False))
        else:
            self.memory_proj = nn.Identity()          # never called, but keeps code simple

        self.device = device
    # ----------------------------------------------------------------------

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers,
                          n_neighbors=20, time_diffs=None, use_time_proj=True):
        """
        Recursive temporal-neighbour encoder.
        """

        assert n_layers >= 0
        src_t = torch.from_numpy(source_nodes).long().to(self.device)
        ts_t  = torch.from_numpy(timestamps).float().to(self.device).unsqueeze(1)

        # Δt=0 for the “query” node itself
        src_time_emb = self.time_proj(self.time_encoder(torch.zeros_like(ts_t)))

        h_src = self.node_proj(self.node_features[src_t])

        if self.use_memory:
            mem_tensor = self._tensor_from_memory(memory)            # <-- NEW
            h_mem = self.memory_proj(mem_tensor[source_nodes])       # shape [B, D]
            h_src = h_src + h_mem

        if n_layers == 0:
            return h_src                                         # base case
        # ------------------------------------------------------------------

        # ---------- 2. recurse one layer down ------------------------------
        h_src_conv = self.compute_embedding(memory, source_nodes, timestamps,
                                            n_layers=n_layers - 1,
                                            n_neighbors=n_neighbors)
        # ------------------------------------------------------------------

        # ---------- 3. fetch neighbours ------------------------------------
        neigh, eidx, etime = self.neighbor_finder.get_temporal_neighbor(
                                   source_nodes, timestamps, n_neighbors)
        neigh_t   = torch.from_numpy(neigh).long().to(self.device)
        eidx_t    = torch.from_numpy(eidx ).long().to(self.device)

        dt        = timestamps[:, None] - etime                  # [B, k]
        dt_t      = torch.from_numpy(dt).float().to(self.device)
        dt_emb    = self.time_proj(self.time_encoder(dt_t))      # [B, k, D]

        neigh_flat   = neigh.flatten()
        neigh_emb    = self.compute_embedding(memory,
                            neigh_flat,
                            np.repeat(timestamps, n_neighbors),
                            n_layers=n_layers - 1,
                            n_neighbors=n_neighbors)
        k = n_neighbors if n_neighbors > 0 else 1
        neigh_emb = neigh_emb.view(len(source_nodes), k, -1)     # [B, k, D]

        edge_emb   = self.edge_proj(self.edge_features[eidx_t])  # [B, k, D]
        mask       = neigh_t == 0
        # ------------------------------------------------------------------

        # ---------- 4. aggregate in chosen subclass ------------------------
        out = self.aggregate(n_layers,
                             h_src_conv,            # source conv emb
                             src_time_emb,          # [B, 1, D] after squeeze in aggregate
                             neigh_emb,             # [B, k, D]
                             dt_emb,                # [B, k, D]
                             edge_emb,              # [B, k, D]
                             mask)

        return out
    # ----------------------------------------------------------------------

    # subclasses implement attention / sum here
    def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, edge_features, mask):
        raise NotImplementedError
    
    def _tensor_from_memory(self, memory_obj_or_tensor):
      """
      Accepts a Memory object *or* its inner tensor and
      returns the [N, mem_dim] tensor.
      """
      if isinstance(memory_obj_or_tensor, torch.Tensor):
          return memory_obj_or_tensor            # already the tensor
      else:
          return memory_obj_or_tensor.memory     # unwrap from Memory object



class GraphSumEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    
    super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                            edge_features=edge_features,
                                            memory=memory,
                                            neighbor_finder=neighbor_finder,
                                            time_encoder=time_encoder, n_layers=n_layers,
                                            n_node_features=n_node_features,
                                            n_edge_features=n_edge_features,
                                            n_time_features=n_time_features,
                                            embedding_dimension=embedding_dimension,
                                            device=device,
                                            n_heads=n_heads, dropout=dropout,
                                            use_memory=use_memory)
    # Creating the first layer of the netwrok, applies the basic linear transformation. 
    # input being the embedding dim + number of time features + number of edge features
    # output being the embedding dimention (ready for the decoder later, again this is the embedding modules so this makes sense)
    # Now it does this, adds these linear transformation layers for the number of layers 
    # self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features + n_edge_features, embedding_dimension)
    #                                      for _ in range(n_layers)])
    # # this is the second layer ??
    # self.linear_2 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_node_features + n_time_features, embedding_dimension) 
    #                                      for _ in range(n_layers)])
    
    D = embedding_dimension          # shorthand

    # neighbours ⊕ edge ⊕ time   →   D
    self.linear_1 = nn.ModuleList([
        nn.Linear(3 * D, D)          # 3·D  →  D
        for _ in range(n_layers)
    ])

    # neighbours_sum ⊕ src ⊕ time  →  D
    self.linear_2 = nn.ModuleList([
        nn.Linear(3 * D, D)          # 3·D  →  D
        for _ in range(n_layers)
    ])

  # Aggregation -- which aggregates the information to produce a source embedding
  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    # concatenate the neighbor embeddings, edge time embeddings and edge features -- very basic 
    # this is just to actually concatenation / group the information about the neighbourhood into a form that can be embedded
    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features], dim=2)
    # now we actually embedd the neighbourhood -- done by passing the concatenated information through the first layer 
    neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
    # Applies the rectified linear unit function element-wise to the sum of the neighbourhood embeddings
    # neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))
    neighbors_mean = torch.nn.functional.relu(torch.mean(neighbor_embeddings, dim=1))

    # now we create a vect of the source's features via concatenation 
    source_features = torch.cat([source_node_features, source_nodes_time_embedding.squeeze()], dim=1)
    # concatenate that with the result of the neighbourhood sum
    # source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
    
    source_embedding = torch.cat([neighbors_mean, source_features], dim=1)
    # then we pass this information through the second layer to get the embedding of the node 
    # and it's neighbourhood (pretty smart tbh) -- this begs the question -- HOW DO WE TRAIN?
    source_embedding = self.linear_2[n_layer - 1](source_embedding)

    return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                                                  n_heads, dropout,  use_memory)

    proj_dim = embedding_dimension          # shorthand: D
    self.attention_models = nn.ModuleList([
        TemporalAttentionLayer(
            n_node_features      = proj_dim,     # <-- was  n_node_features
            n_neighbors_features = proj_dim,     # <-- same
            n_edge_features      = proj_dim,     # <-- was  n_edge_features
            time_dim             = proj_dim,     # <-- was  n_time_features
            n_head               = n_heads,
            dropout              = dropout,
            output_dimension     = proj_dim      # keep output = D
        )
        for _ in range(n_layers)
    ])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    attention_model = self.attention_models[n_layer - 1]

    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)
    source_embedding = torch.nn.functional.layer_norm(
      source_embedding, source_embedding.shape[-1:])
    return source_embedding


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True):
  if module_type == "graph_attention":
    return GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory)
  elif module_type == "graph_sum":
    return GraphSumEmbedding(node_features=node_features,
                              edge_features=edge_features,
                              memory=memory,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory)

  elif module_type == "identity":
    return IdentityEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout)
  elif module_type == "time":
    return TimeEmbedding(node_features=node_features,
                         edge_features=edge_features,
                         memory=memory,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout,
                         n_neighbors=n_neighbors)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))





