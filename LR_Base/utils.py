import torch 
import os 
import pandas as pd 
import numpy as np 
import scanpy as sc
import scipy 
import matplotlib.pyplot as plt 
import torch.nn as nn 
import torch.nn.functional as F 
from torch_geometric.data import HeteroData 
from torch_geometric.nn import GATConv, PNAConv
#from metapath2vec import MetaPath2Vec 
import warnings 
warnings.filterwarnings('ignore')
import math
import gzip
import pickle
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import issparse
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.nn import summary
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter(log_dir='/p300s/xiaojf_group/xuchenle/DLCCC/model/tensorboard')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(path, interactions_path, complex_nodes_path, complex_interactions_path, cutoff=0.8):
    adata = sc.read_h5ad(path)
    LR_interactions = pd.read_csv(interactions_path)
    complex_nodes = pd.read_csv(complex_nodes_path)
    complex_interactions = pd.read_csv(complex_interactions_path)
    complex_interactions_single = complex_interactions.loc[~complex_interactions['Src'].str.contains('&')]['Src'].unique().tolist() + complex_interactions.loc[~complex_interactions['Dst'].str.contains('&')]['Dst'].unique().tolist()

    ligands_nodes = LR_interactions['Src'].unique().tolist()
    receptors_nodes = LR_interactions['Dst'].unique().tolist()
    complex_subunits = complex_nodes['subunit'].unique().tolist()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    highly_variable_genes = adata.var[adata.var['highly_variable']]
    if issparse(adata.X):
        adata.X = adata.X.todense()
    highly_variable_matrix = adata[:, adata.var['highly_variable']].X
    highly_variable_matrix = pd.DataFrame(highly_variable_matrix, index=adata.obs_names, columns=highly_variable_genes.index)
    mean_expression = np.array(np.mean(adata.X, axis=0))

    threshold = np.quantile(mean_expression, cutoff)

    high_expression_genes = adata.var.index[mean_expression[0] > threshold]
    high_expression_matrix = adata[:, high_expression_genes].X
    high_expression_matrix = pd.DataFrame(high_expression_matrix, index=adata.obs_names, columns=high_expression_genes)
    high_expression_matrix = high_expression_matrix.T
    highly_variable_matrix = highly_variable_matrix.T
    matrix = pd.concat([highly_variable_matrix, high_expression_matrix]).drop_duplicates()

    meta = adata.obs['cell_type_original']
    meta = pd.DataFrame({"cell": meta.index.tolist(), "label_name": meta.tolist()})

    nodes = matrix.index.tolist()
    cell_LR_nodes = list(set(ligands_nodes + receptors_nodes) & set(nodes))
    cell_ligand_nodes = list(set(ligands_nodes) & set(nodes))
    cell_receptor_nodes = list(set(receptors_nodes) & set(nodes))
    cell_complex_nodes = list(set(complex_subunits) & set(nodes))
    cell_complex_nodes_single = list(set(complex_interactions_single) & set(nodes))

    celltype = pd.DataFrame({'label': meta['label_name'].unique().tolist(), 'Id': list(range(meta['label_name'].unique().tolist().__len__()))})
    celltype.index = celltype['label'].tolist()
    meta['label'] = celltype.loc[meta['label_name']]['Id'].tolist()
    matrix = matrix.transpose()
  
    group_cells = meta.groupby('label')
    gene_mean = {}
    for label, cells in group_cells:
        gene_mean[label] = matrix[matrix.index.isin(cells['cell'].tolist())].median(axis=0) ## median instead of mean
    gene_mean_df = pd.DataFrame(gene_mean)
    gene_mean_df.dropna(axis=0, inplace=True)
    list_data = []

    for gene in gene_mean_df.index:
        for cell_type in gene_mean_df.columns:
            list_data.append((gene + "_" + str(cell_type), gene_mean_df.loc[gene, cell_type]))

    stacked_df = pd.DataFrame(list_data, columns=["gene", "expression"])


    cell_nodes = stacked_df.loc[stacked_df['expression'] != 0]
    cell_nodes.index = [i.split('_')[0] for i in cell_nodes['gene'].tolist()]


    cell_ligands_df = cell_nodes.loc[cell_nodes.index.isin(cell_ligand_nodes)]
    cell_receptors_df = cell_nodes.loc[cell_nodes.index.isin(cell_receptor_nodes)]
    cell_ligands_df['identifier'] = [ i + '_Ligand' for i in cell_ligands_df['gene'].tolist()]
    cell_receptors_df['identifier'] = [ i + '_Receptor' for i in cell_receptors_df['gene'].tolist()]
    cell_ligands_df['gene'] = cell_ligands_df.index.tolist()
    cell_receptors_df['gene'] = cell_receptors_df.index.tolist()
    cell_LR_nodes = pd.concat([cell_ligands_df, cell_receptors_df])
    cell_LR_nodes['category'] = [i.split('_')[2] for i in cell_LR_nodes['identifier']]
    cell_LR_nodes['cellgroup'] = [i.split('_')[1] for i in cell_LR_nodes['identifier']]
    cell_LR_nodes.index = range(0, cell_LR_nodes.shape[0])

    # Complex
    cell_complex_df = cell_nodes.loc[cell_nodes.index.isin(cell_complex_nodes)]
    cell_complex_df['cellgroup'] = [i.split('_')[1] for i in cell_complex_df['gene']]
    cell_complex_df['gene'] = cell_complex_df.index.tolist()
    valid_complexes_dict = filter_complex_nodes(cell_LR_nodes, complex_nodes)

    complex_expr_df = calculate_complex_expression(cell_complex_df, valid_complexes_dict, complex_nodes)

    cell_LR_interactions = LR_interactions[LR_interactions['Src'].isin(cell_ligand_nodes) & LR_interactions['Dst'].isin(cell_receptor_nodes)]
    Src_nodes = cell_LR_interactions['Src'].unique().tolist()
    Dst_nodes = cell_LR_interactions['Dst'].unique().tolist()

    cell_ligands_df = cell_ligands_df.loc[cell_ligands_df['gene'].isin(Src_nodes)]
    cell_receptors_df = cell_receptors_df.loc[cell_receptors_df['gene'].isin(Dst_nodes)]

    cell_LR_nodes = pd.concat([cell_ligands_df, cell_receptors_df])
    cell_LR_nodes['category'] = [i.split('_')[2] for i in cell_LR_nodes['identifier']]
    cell_LR_nodes['cellgroup'] = [i.split('_')[1] for i in cell_LR_nodes['identifier']]

    if not complex_expr_df.empty:
        cell_LR_nodes = pd.concat([cell_LR_nodes, complex_expr_df])
        cell_complexes = complex_expr_df['gene'].unique().tolist()
        cell_complexes_extend = list(set(cell_complexes + cell_LR_nodes['gene'].unique().tolist() + cell_complex_nodes_single))
        cell_complex_interactions = complex_interactions[complex_interactions['Src'].isin(cell_complexes_extend) & complex_interactions['Dst'].isin(cell_complexes_extend)]
        cell_LR_interactions = pd.concat([cell_LR_interactions, cell_complex_interactions])
    cell_LR_nodes.index = range(cell_LR_nodes.shape[0])
    cell_LR_interactions.index = range(cell_LR_interactions.shape[0])
    return cell_LR_nodes, cell_LR_interactions, meta, celltype

def filter_complex_nodes(cell_LR_nodes, complex_nodes):
    complex_dict = complex_nodes.groupby('complex')['subunit'].apply(list).to_dict()
    valid_complexes_dict = {}
    for cellgroup in cell_LR_nodes['cellgroup'].unique():
        cell_LR_nodes_cellgroup = cell_LR_nodes[cell_LR_nodes['cellgroup'] == cellgroup]
        cell_genes = cell_LR_nodes_cellgroup['gene'].unique().tolist()
        filter_complex_nodes = complex_nodes[complex_nodes['subunit'].isin(cell_genes)]
        filter_complex_dict = filter_complex_nodes.groupby('complex')['subunit'].apply(list).to_dict()
        valid_complexes = []
        for complex in filter_complex_dict.keys():
            if complex_dict[complex] == filter_complex_dict[complex]:
                valid_complexes.append(complex)
        valid_complexes_dict[cellgroup] = valid_complexes
    return valid_complexes_dict

def calculate_complex_expression(cell_complex_df, valid_complexes_dict, complex_nodes):
    if all(not v for v in valid_complexes_dict.values()):
        return 'No valid complexes found for the given cell groups'
    else:
        complex_expr = {}
        for cellgroup in cell_complex_df['cellgroup'].unique():
            complex_expr[cellgroup] = {}
            cell_nodes_cellgroup = cell_complex_df[cell_complex_df['cellgroup'] == cellgroup]
            for i in valid_complexes_dict[cellgroup]:
                complex_expr[cellgroup][i] = []
                for j in i.split('&'):
                    complex_expr[cellgroup][i].append(cell_nodes_cellgroup[cell_nodes_cellgroup['gene'] == j]['expression'].values[0])
        complex_expr_df = []
        for i in complex_expr.keys():
            for j in complex_expr[i].keys():
                complex_expr_df.append([j, i, np.min(complex_expr[i][j])])
        complex_expr_df = pd.DataFrame(complex_expr_df, columns=['gene', 'cellgroup', 'expression'])
        complex_expr_df['category'] = complex_expr_df['gene'].apply(lambda x: list(complex_nodes.loc[complex_nodes['complex'] == x]['category'].values))
        complex_expr_df = complex_expr_df.explode('category')
        complex_expr_df['identifier'] = complex_expr_df['gene'] + '_' + complex_expr_df['cellgroup'] + '_' + complex_expr_df['category']
        complex_expr_df = complex_expr_df.drop_duplicates(subset=['identifier'])
        return complex_expr_df



def get_full_matrix(cell_LR_nodes, cell_LR_interactions):
    cell_ligand_df = cell_LR_nodes.loc[cell_LR_nodes['category']=='Ligand']
    cell_receptor_df = cell_LR_nodes.loc[cell_LR_nodes['category']=='Receptor']
    filtered_interactions = cell_LR_interactions[cell_LR_interactions['Src'].isin(cell_ligand_df['gene']) & cell_LR_interactions['Dst'].isin(cell_receptor_df['gene'])]

    ligands_df = cell_ligand_df[['gene', 'identifier']]
    receptors_df = cell_receptor_df[['gene', 'identifier']]
    merged_df = pd.merge(filtered_interactions, ligands_df, left_on='Src', right_on='gene', how='left')
    merged_df['Src'] = merged_df['identifier']
    # 删除 'gene' 和 'identifier' 列
    merged_df = merged_df.drop(columns=['gene', 'identifier'])
    # 最后，将 genes_df 与 filter_interactions 的 'receptor' 列进行合并
    merged_df = pd.merge(merged_df, receptors_df, left_on='Dst', right_on='gene', how='left')
    # 将 'receptor' 列替换为 'identifier'
    merged_df['Dst'] = merged_df['identifier']
    # 删除 'gene' 和 'identifier' 列
    merged_df = merged_df.drop(columns=['gene', 'identifier'])
    merged_df.drop_duplicates(subset=['Src', 'Dst'], inplace=True)
    merged_df['Src_group'] = [i.split('_')[1] for i in merged_df['Src']]
    merged_df['Dst_group'] = [i.split('_')[1] for i in merged_df['Dst']]
    merged_df = merged_df[merged_df['Src_group'] != merged_df['Dst_group']]
    cell_ligand_df.index = cell_ligand_df['identifier'].tolist()
    cell_receptor_df.index = cell_receptor_df['identifier'].tolist()
    merged_df['Src_expression'] = cell_ligand_df.loc[merged_df['Src'], 'expression'].values
    merged_df['Dst_expression'] = cell_receptor_df.loc[merged_df['Dst'], 'expression'].values
    merged_df['expression'] = cell_ligand_df.loc[merged_df['Src'], 'expression'].values * cell_receptor_df.loc[merged_df['Dst'], 'expression'].values

    nodes = pd.DataFrame({'identifier':cell_LR_nodes['identifier'].tolist()})
    
    nodes['Group'] = [int(i.split('_')[1]) for i in nodes['identifier']]
    nodes['expression'] = cell_LR_nodes['expression'].tolist()
    nodes['Id'] = nodes.groupby('Group').cumcount()
    interactions = merged_df[['Src', 'Src_expression', 'Dst', 'Dst_expression', 'Src_group', 'Dst_group']]
    interactions['LR'] = interactions.apply(lambda row: f"{row['Src'].split('_')[0]}_{row['Dst'].split('_')[0]}", axis=1)
    interactions['Src'] = merged_df['Src'].map(nodes.set_index('identifier')['Id'])
    interactions['Dst'] = merged_df['Dst'].map(nodes.set_index('identifier')['Id'])
    interactions['expression'] = merged_df['expression']
    interactions['label'] = interactions.apply(lambda row: f"{row['Src_group']}_to_{row['Dst_group']}", axis=1)
    interactions['LR_id'] = [(i, j) for i, j in zip(interactions['Src'], interactions['Dst'])]
    interactions = interactions.groupby('label', as_index=False).apply(lambda x: x.sort_values('LR_id'))
    interactions.index = range(interactions.shape[0])
    nodes.index = range(nodes.shape[0])

    nodes['category'] = [i.split('_')[2] for i in nodes['identifier']]
    merged_df.index = range(merged_df.shape[0])

    return merged_df, interactions, nodes

def replace_ids(nodes, interactions):
    # 将nodes数据框的索引设置为旧的ID
    nodes = nodes.set_index('Id')

    # 将interactions数据框的'Src'列与nodes数据框进行合并，以获取新的ID
    interactions = interactions.merge(nodes['NewID'], left_on='Src', right_index=True)

    # 将新的ID列的名称改为'Src'
    interactions = interactions.rename(columns={'NewID': 'Src_New'})

    # 将interactions数据框的'Dst'列与nodes数据框进行合并，以获取新的ID
    interactions = interactions.merge(nodes['NewID'], left_on='Dst', right_index=True)

    # 将新的ID列的名称改为'Dst'
    interactions = interactions.rename(columns={'NewID': 'Dst_New'})
    interactions = interactions[['Src_New', 'Dst_New', 'expression', 'label']]
    return interactions

def make_dataset(nodes, interactions):
    #interactions = replace_ids(nodes, interactions)
    # 将节点表达值转换为张量
    x = torch.tensor(nodes['expression'].values, dtype=torch.float).view(-1, 1)
    y = torch.tensor(nodes['Id'].values, dtype=torch.long)
    # 将互作关系转换为张量
    #edge_index = torch.tensor(interactions[['Src', 'Dst']].values.T, dtype=torch.long)
    # 创建 HeteroData 对象
    data = HeteroData()
    # 添加节点信息
    for group in nodes['Group'].unique():
        group_nodes = nodes[nodes['Group'] == group]
        data[str(group)] = {'x': x[group_nodes.index], 'y': y[group_nodes.index], 'num_nodes': group_nodes.shape[0]}
    
    edge_index_dict = {}
    edge_attr_dict = {}
    # 遍历所有可能的 Group 组合
    for interaction_type in interactions['label'].unique():
        src_group = interaction_type.split('_')[0]
        dst_group = interaction_type.split('_')[-1]
        filtered_interaction = interactions[interactions['label'] == interaction_type]
        # 找到所有从 src_group 到 dst_group 的边
        edges = filtered_interaction[(filtered_interaction['Src'].isin(nodes[nodes['Group'] == int(src_group)]['Id'])) &
                            (filtered_interaction['Dst'].isin(nodes[nodes['Group'] == int(dst_group)]['Id']))]
        edges.index = range(edges.shape[0])
        # 如果有这样的边，将它们添加到 edge_index_dict
        if not edges.empty:
            edge_index = torch.tensor(edges[['Src', 'Dst']].values.T, dtype=torch.long)
            edge_attr = torch.tensor(edges['expression'].values, dtype=torch.float).view(-1, 1)
            edge_index_dict[(src_group, src_group+'_to_'+dst_group, dst_group)] = edge_index
            edge_attr_dict[(src_group, src_group+'_to_'+dst_group, dst_group)] = edge_attr
    # 将 edge_index_dict 添加到 HeteroData 对象
    data.edge_index_dict = edge_index_dict
    data.edge_attr_dict = edge_attr_dict
    
    return data

def make_metapath(celltype):
    metapath = []
    celltype_id = [str(i) for i in celltype['Id'].tolist()] * 2
    for i in range(celltype['Id'].max() + 1):
        path = (celltype_id[i], celltype_id[i]+ '_to_' + celltype_id[i+1], celltype_id[i+1])
        metapath.append(path)
    return metapath



def get_node_embedding(nodes, model):
    embedding_dict = {}
    celltype = nodes['Group'].unique().tolist()
    for i in celltype:
        embedding_dict[str(celltype[i])] = model(str(celltype[i])).to('cpu').detach().numpy()

    return embedding_dict

def get_similarity_dict(nodes, embedding_dict, edge_index_dict):
    ligands_num_dict = {}
    similarity_dict = {}
    receptors_num_dict = {}
    scaler = MinMaxScaler()

    for i in nodes['Group'].unique():
        ligands_num_dict[str(i)] = nodes.loc[(nodes['Group'] == i) & (nodes['category'] == 'Ligand')].shape[0]
        receptors_num_dict[str(i)] = nodes.loc[(nodes['Group'] == i) & (nodes['category'] == 'Receptor')].shape[0]

    for i in edge_index_dict.keys():
        similarity_dict[i] = np.dot(embedding_dict[i[0]][: ligands_num_dict[i[0]]], embedding_dict[i[-1]][ligands_num_dict[i[-1]]: ].T)
    similarity_lst_dict = {}
    for i in similarity_dict.keys():
        similarity_lst = []
        for j, k in zip(edge_index_dict[i][0], edge_index_dict[i][1]):
            similarity_lst.append(similarity_dict[i][int(j), int(k)-ligands_num_dict[i[-1]]])
        similarity_lst = np.array(similarity_lst).reshape(-1, 1)
        similarity_lst_dict[i] = scaler.fit_transform(similarity_lst)
    return similarity_lst_dict

def get_communication_dict(similarity_lst_dict, edge_index_dict):
    edge_index_set_dict = {}
    for i in edge_index_dict.keys():
        edge_index_set = []
        for j in range(edge_index_dict[i].shape[1]):
            edge_index_set.append((int(edge_index_dict[i][0][j]), int(edge_index_dict[i][1][j])))
        edge_index_set_dict[i] = edge_index_set
    indices_dict = {}
    for i in similarity_lst_dict.keys():
        mean = np.mean(similarity_lst_dict[i])
        indices = np.where(similarity_lst_dict[i] > mean)
        indices_dict[i] = indices[0]
    communication_dict = {}
    for i in indices_dict.keys():
        communication_dict[i] = {}
        indices = indices_dict[i]
        communication = [edge_index_set_dict[i][index] for index in indices_dict[i]]
        for j, k in zip(communication, indices):
            communication_dict[i][j] = similarity_lst_dict[i][k] 
    return communication_dict

def get_communication_df(communication_dict, interactions):
    communication_df_dict = {}
    for i in communication_dict.keys():
        communication_df = pd.DataFrame(communication_dict[i].items(), columns=['LR_id', 'communication_score']).sort_values(by='communication_score', ascending=False)
        label = i[1]
        filter_interactions = interactions[interactions['label'] == label]
        communication_df = pd.merge(filter_interactions, communication_df, on='LR_id', how='left')
        communication_df = communication_df.dropna(subset=['communication_score'])
        communication_df['communication_score'] = communication_df['communication_score'].apply(lambda x: x.item())
        communication_df_dict[i] = communication_df.sort_values(by='communication_score', ascending=False)
    return communication_df_dict

    

def gene_mean_process(matrix, meta, id):
    group_cells = meta.groupby(id)
    gene_mean = {}
    if meta['cell'].tolist()[0] not in matrix.index.tolist():
        matrix = matrix.transpose()
    for label, cells in group_cells:
        gene_mean[label] = matrix[matrix.index.isin(cells['cell'].tolist())].mean(axis=0)

    return pd.DataFrame(gene_mean)

def save_data(data, path):
    with gzip.open(path, 'wb') as f:
        pickle.dump(data, f)

def load_data_2(path):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_distance(coord1, coord2):
    # Unpack coordinates
    x1, y1 = coord1
    x2, y2 = coord2
    
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) **2)


def calculate_spatial_distance(adata, cell_type ):
    centroids = adata.obs.groupby(cell_type)[['x_coord', 'y_coord']].mean()
    distances = pd.DataFrame(index=centroids.index, columns=centroids.index)
    for cell_type1 in centroids.index:
        for cell_type2 in centroids.index:
            distances.loc[cell_type1, cell_type2] = calculate_distance(centroids.loc[cell_type1], centroids.loc[cell_type2])
    return distances    




