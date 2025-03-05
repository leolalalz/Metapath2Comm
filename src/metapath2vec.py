from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch_geometric.utils.sparse import index2ptr
from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import sort_edge_index

EPS = 1e-15


class MetaPath2Vec(torch.nn.Module):
    def __init__(
        self,
        edge_index_dict: Dict[EdgeType, Tensor],
        embedding_dim: int,
        metapath: List[EdgeType],
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        num_nodes_dict: Optional[Dict[NodeType, int]] = None,
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
        sparse: bool = False,
    ):
        super().__init__()

        if num_nodes_dict is None:
            num_nodes_dict = {}
            for keys, edge_index in edge_index_dict.items():
                key = keys[0]
                N = int(edge_index[0].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

                key = keys[-1]
                N = int(edge_index[1].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))
        
        self.edge_attr_dict = edge_attr_dict

        self.rowptr_dict, self.col_dict, self.rowcount_dict = {}, {}, {}
        for keys, edge_index in edge_index_dict.items():
            sizes = (num_nodes_dict[keys[0]], num_nodes_dict[keys[-1]])
            row, col = sort_edge_index(edge_index, num_nodes=max(sizes)).cpu()
            rowptr = index2ptr(row, size=sizes[0])
            self.rowptr_dict[keys] = rowptr
            self.col_dict[keys] = col
            self.rowcount_dict[keys] = rowptr[1:] - rowptr[:-1]

        for edge_type1, edge_type2 in zip(metapath[:-1], metapath[1:]):
            if edge_type1[-1] != edge_type2[0]:
                raise ValueError(
                    "Found invalid metapath. Ensure that the destination node "
                    "type matches with the source node type across all "
                    "consecutive edge types.")

        assert walk_length + 1 >= context_size
        if walk_length > len(metapath) and metapath[0][0] != metapath[-1][-1]:
            raise AttributeError(
                "The 'walk_length' is longer than the given 'metapath', but "
                "the 'metapath' does not denote a cycle")

        self.embedding_dim = embedding_dim
        self.metapath = metapath
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.num_nodes_dict = num_nodes_dict

        types = set([x[0] for x in metapath]) | set([x[-1] for x in metapath])
        types = sorted(list(types))

        count = 0
        self.start, self.end = {}, {}
        for key in types:
            self.start[key] = count
            count += num_nodes_dict[key]
            self.end[key] = count

        offset = [self.start[metapath[0][0]]]
        offset += [self.start[keys[-1]] for keys in metapath
                   ] * int((walk_length / len(metapath)) + 1)
        offset = offset[:walk_length + 1]
        assert len(offset) == walk_length + 1
        self.offset = torch.tensor(offset)

        # + 1 denotes a dummy node used to link to for isolated nodes.
        self.embedding = Embedding(count + 1, embedding_dim, sparse=sparse)
        self.dummy_idx = count

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, node_type: str, batch: OptTensor = None) -> Tensor:
        emb = self.embedding.weight[self.start[node_type]:self.end[node_type]]
        return emb if batch is None else emb.index_select(0, batch)

    def loader(self, **kwargs):
        return DataLoader(range(self.num_nodes_dict[self.metapath[0][0]]),
                          collate_fn=self._sample, **kwargs)

    def _pos_sample(self, batch: Tensor) -> Tensor:
        
        batch = batch.repeat(self.walks_per_node)

        rws = [batch]
        for i in range(self.walk_length):
            edge_type = self.metapath[i % len(self.metapath)]
            batch = sample(
                self.rowptr_dict[edge_type],
                self.col_dict[edge_type],
                self.rowcount_dict[edge_type],
                batch,
                num_neighbors=1,
                dummy_idx=self.dummy_idx,
                edge_attr=self.edge_attr_dict[edge_type]
            ).view(-1)
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))
        rw[rw > self.dummy_idx] = self.dummy_idx

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def _neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            batch = torch.randint(0, self.num_nodes_dict[keys[-1]],
                                  (batch.size(0), ), dtype=torch.long)
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def _sample(self, batch: List[int]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch, dtype=torch.long)
        return self._pos_sample(batch), self._neg_sample(batch)

    
    def loss(self, pos_rw: torch.Tensor, neg_rw: torch.Tensor) -> torch.Tensor:
        pos_start, pos_rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        neg_start, neg_rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_pos_start = self.embedding(pos_start).view(pos_rw.size(0), 1, self.embedding_dim)
        h_pos_rest = self.embedding(pos_rest.view(-1)).view(pos_rw.size(0), -1, self.embedding_dim)

        h_neg_start = self.embedding(neg_start).view(neg_rw.size(0), 1, self.embedding_dim)
        h_neg_rest = self.embedding(neg_rest.view(-1)).view(neg_rw.size(0), -1, self.embedding_dim)

        pos_score = (h_pos_start * h_pos_rest).sum(dim=-1).view(-1)  # [batch_size]
        neg_score = (h_neg_start * h_neg_rest).sum(dim=-1).view(-1)  # [batch_size]

        pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score), reduction='mean')
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score), reduction='mean')

        return pos_loss + neg_loss


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'{self.embedding.weight.size(0) - 1}, '
                f'{self.embedding.weight.size(1)})')


def sample(rowptr: torch.Tensor, col: torch.Tensor, rowcount: torch.Tensor, 
           subset: torch.Tensor, num_neighbors: int, dummy_idx: int, 
           edge_attr: torch.Tensor) -> torch.Tensor:
    mask = subset >= dummy_idx
    subset = subset.clamp(min=0, max=rowptr.numel() - 2)
    count = rowcount[subset]

    rand = torch.full((subset.size(0), num_neighbors), dummy_idx, dtype=torch.long, device=rowptr.device)

    valid = count > 0
    valid_subset = subset[valid]
    valid_count = count[valid]

    if valid_count.numel() == 0:
        rand[mask.repeat_interleave(num_neighbors)] = dummy_idx
        return col[rand.clamp(max=col.numel() - 1)] if col.numel() > 0 else rand

    start = rowptr[valid_subset]
    end = start + valid_count

    max_count = valid_count.max()
    weights_matrix = torch.zeros((valid_subset.size(0), max_count), dtype=edge_attr.dtype, device=edge_attr.device)
    for i, (s, e) in enumerate(zip(start, end)):
        weights_matrix[i, :valid_count[i]] = edge_attr[s:e].flatten()

    prob = (weights_matrix / weights_matrix.sum(dim=1, keepdim=True)).nan_to_num(0.0) #处理weights.sum()为0的情况
    sampled_indices = torch.multinomial(prob, num_neighbors, replacement=True)

    row_indices = torch.arange(valid_subset.size(0), device=start.device).unsqueeze(1).repeat(1, num_neighbors)
    absolute_indices = start.unsqueeze(1) + sampled_indices

    rand[valid] = absolute_indices

    rand = rand.clamp(max=col.numel() - 1)
    col_sampled = col[rand] if col.numel() > 0 else rand
    col_sampled[mask.repeat_interleave(num_neighbors) | (count == 0).repeat_interleave(num_neighbors)] = dummy_idx # count == 0的情况也需要处理

    return col_sampled