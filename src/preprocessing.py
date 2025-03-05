import argparse
import pandas as pd
import numpy as np
import scanpy as sc
import torch
from scipy.spatial import distance_matrix
from anndata import AnnData
from torch_geometric.data import HeteroData
import os
import pickle
import gzip
import warnings
import sys
warnings.filterwarnings('ignore')   


def get_gene_expression(adata, LR_nodes, deg=True, prop=0.1, celltype='cell_type'):   
    adata = adata[:, adata.var.index.isin(LR_nodes)].copy()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    cell_type_counts = adata.obs[celltype].value_counts()
    cell_types_to_keep = cell_type_counts[cell_type_counts >= 2].index
    adata = adata[adata.obs[celltype].isin(cell_types_to_keep)].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cell_type_counts = adata.obs[celltype].value_counts()
    cell_type_counts = pd.DataFrame(cell_type_counts.values, index=cell_type_counts.index.tolist(), columns=['count'])

    if adata.var.columns.str.contains('feature_name').any():
        marker_genes = adata.var['feature_name'].tolist()
    else:
        marker_genes = adata.var.index.tolist()

    cell_types = adata.obs[celltype]
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()
    gene_expression = adata.X

    if deg:
        # degs
        sc.tl.rank_genes_groups(adata, celltype, method='t-test')
        result = adata.uns['rank_genes_groups']
        groups = result['names'].dtype.names
        degs = pd.DataFrame(
            {group + '_' + key: result[key][group]
            for group in groups for key in ['names', 'pvals', 'logfoldchanges']}
        )
        deg_df = pd.DataFrame(columns=['cell_type', 'gene', 'pval', 'logfoldchange'])
        for i in groups:
            degg = degs.loc[(degs[i+'_pvals'] < 0.05)]
            degg = degg[[i+'_names', i+'_pvals', i+'_logfoldchanges']]
            degg.columns = ['gene', 'pval', 'logfoldchange']
            degg['cell_type'] = i
            deg_df = pd.concat([deg_df, degg])
        if deg_df.empty:
            print("deg_df is empty, please check the input data.")
            sys.exit(1)
        if (deg_df['logfoldchange'] > 0.5).sum() == 0:
            print("No DEG found, please check the input data.")
            sys.exit(1)
        deg_df['identifier'] = deg_df['gene'] + '_' + deg_df['cell_type']
        deg_df.loc[deg_df['logfoldchange'] > 0.5, 'DEG'] = 'True'
        deg_df = deg_df.loc[deg_df['DEG'] == 'True']
        marker_genes = deg_df['gene'].unique().tolist()
        gene_expression = adata[:, marker_genes].X.toarray()

    celltype_expression = pd.DataFrame(gene_expression, index=cell_types, columns=marker_genes)
    proportion = celltype_expression.groupby(level=0)[marker_genes].apply(lambda x: (x > 0).mean()).T
    celltype_expression = celltype_expression.groupby(level=0).mean()
    gene_expression_df = celltype_expression.stack().to_frame(name='expression').reset_index()
    gene_expression_df = gene_expression_df.loc[gene_expression_df['expression'] != 0]
    gene_expression_df.columns = ['celltype', 'gene', 'expression']

    gene_expression_df['identifier'] = gene_expression_df['gene'].astype(str) + '_' + gene_expression_df['celltype'].astype(str)
    gene_expression_df['gene'] = [i.split('_')[0] for i in gene_expression_df['identifier']]
    gene_expression_df['celltype'] = [i.split('_')[1] for i in gene_expression_df['identifier']]
    cell_LR_nodes = gene_expression_df
    props = []
    for index, row in cell_LR_nodes.iterrows():
        props.append(proportion.loc[row['gene'], row['celltype']])
    cell_LR_nodes['props'] = props
    
    if deg:
        cell_LR_nodes = cell_LR_nodes.loc[cell_LR_nodes['identifier'].isin(deg_df['identifier'])]

    cell_LR_nodes = cell_LR_nodes.loc[cell_LR_nodes['props'] >= prop]
    cell_LR_nodes = cell_LR_nodes.sort_values(by='expression', ascending=False).reset_index(drop=True)
    cell_LR_nodes = cell_LR_nodes.drop(columns=['identifier'])

    return cell_LR_nodes

def calculate_complex_expr(complex_nodes, cell_nodes_exp):
    ligand_complex_dict = {}
    for i in complex_nodes.loc[complex_nodes['category'] == 'Ligand']['complex'].unique():
        ligand_complex_dict[i] = i.split('&')

    receptor_complex_dict = {}
    for i in complex_nodes.loc[complex_nodes['category'] == 'Receptor']['complex'].unique():
        receptor_complex_dict[i] = i.split('&')

    ligand_complex_genes = []
    receptor_complex_genes = []

    for celltype in cell_nodes_exp['celltype'].unique():
        cell_nodes_exp_celltype = cell_nodes_exp.loc[cell_nodes_exp['celltype'] == celltype]
        cell_nodes_exp_celltype.index = cell_nodes_exp_celltype['gene']
        for complex_name, genes in ligand_complex_dict.items():
            valid_genes = [(gene in cell_nodes_exp_celltype['gene']) for gene in genes]
            if all(valid_genes):
                complex_expr = cell_nodes_exp_celltype.loc[genes, 'expression']
                min_expr_gene = complex_expr.idxmin()
                min_expr_value = complex_expr[min_expr_gene]
                min_expr_rate = cell_nodes_exp_celltype.loc[min_expr_gene]['props']
                
                ligand_complex_genes.append({
                    'gene': complex_name,
                    'celltype': celltype,
                    'category': 'Ligand',
                    'expression': min_expr_value,
                    'props': min_expr_rate
                })

        for complex_name, genes in receptor_complex_dict.items():
            valid_genes = [(gene in cell_nodes_exp_celltype['gene']) for gene in genes]
            if all(valid_genes):

                complex_expr = cell_nodes_exp_celltype.loc[genes, 'expression']

                min_expr_gene = complex_expr.idxmin()

                min_expr_value = complex_expr[min_expr_gene]
                min_expr_rate = cell_nodes_exp_celltype.loc[min_expr_gene]['props']
                
                receptor_complex_genes.append({
                    'gene': complex_name,
                    'celltype': celltype,
                    'category': 'Receptor',
                    'expression': min_expr_value,
                    'props': min_expr_rate
                })

    ligand_complex_df = pd.DataFrame(ligand_complex_genes)
    receptor_complex_df = pd.DataFrame(receptor_complex_genes)
    complex_expr_df = pd.concat([ligand_complex_df, receptor_complex_df], axis=0)
    #complex_expr_df['identifier'] = complex_expr_df['gene'] + '_' + complex_expr_df['celltype'] + '_' + complex_expr_df['category']

    return complex_expr_df

def get_interactions(cell_nodes_exp, complex_expr_df, LR_interactions=None, rec2TF=None):
    cell_nodes_exp = pd.concat([cell_nodes_exp[['gene', 'celltype', 'expression', 'props']], complex_expr_df])
    
    if LR_interactions is not None:
        interactions = LR_interactions.loc[(LR_interactions['ligand'].isin(cell_nodes_exp['gene'])) & (LR_interactions['receptor'].isin(cell_nodes_exp['gene']))]
    if rec2TF is not None:
        interactions = rec2TF.loc[(rec2TF['ligand'].isin(cell_nodes_exp['gene'])) & (rec2TF['receptor'].isin(cell_nodes_exp['gene'])) & (rec2TF['TF'].isin(cell_nodes_exp['gene']))]
        interactions = pd.merge(interactions, cell_nodes_exp, left_on='TF', right_on='gene', how='left')
        interactions = interactions.rename(columns={'expression': 'TF_expression', 'props': 'TF_props', 'celltype': 'TF_celltype'})
    
    interactions = pd.merge(interactions, cell_nodes_exp, left_on='ligand', right_on='gene', how='left')
    interactions = interactions.rename(columns={'expression': 'ligand_expression', 'props': 'ligand_props', 'celltype': 'ligand_celltype'})
    interactions = pd.merge(interactions, cell_nodes_exp, left_on='receptor', right_on='gene', how='left')
    interactions = interactions.rename(columns={'expression': 'receptor_expression', 'props': 'receptor_props', 'celltype': 'receptor_celltype'})
    interactions['LR_expression'] = interactions['ligand_expression'] * interactions['receptor_expression']
    if rec2TF is not None:
        interactions = interactions.loc[interactions['receptor_celltype'] == interactions['TF_celltype']]
    

    return interactions

def create_id_dict(interactions: pd.DataFrame, TF=False):
    celltypes = list(set(interactions['ligand_celltype'].unique().tolist() + interactions['receptor_celltype'].unique().tolist()))
    ligands = interactions['ligand'].unique().tolist()
    receptors = interactions['receptor'].unique().tolist()
    
    celltypes_id = pd.DataFrame({'celltypes': sorted(celltypes), 'id': range(len(celltypes))})
    ligands_id = pd.DataFrame({'ligands': sorted(ligands), 'id': range(len(ligands))})
    receptors_id = pd.DataFrame({'receptors': sorted(receptors), 'id': range(len(receptors))})
    id_dict = {'ligand': ligands_id, 'receptor': receptors_id, 'cell': celltypes_id, 'sourcecell': celltypes_id, 'targetcell': celltypes_id}
    if TF:
        TFs = interactions['TF'].unique().tolist()
        TFs_id = pd.DataFrame({'TFs': sorted(TFs), 'id': range(len(TFs))})
        id_dict['TF'] = TFs_id

    return id_dict

def transfer_id(interactions, id_dict, TF=False):
    celltypes_id = id_dict['cell']
    ligands_id = id_dict['ligand']
    receptors_id = id_dict['receptor']
    interactions['ligand_id'] = interactions['ligand'].map(ligands_id.set_index('ligands')['id'])
    interactions['receptor_id'] = interactions['receptor'].map(receptors_id.set_index('receptors')['id'])
    interactions['sourcecelltype_id'] = interactions['ligand_celltype'].map(celltypes_id.set_index('celltypes')['id'])
    interactions['targetcelltype_id'] = interactions['receptor_celltype'].map(celltypes_id.set_index('celltypes')['id'])   
    if TF:
        TFs_id = id_dict['TF']
        interactions['TF_id'] = interactions['TF'].map(TFs_id.set_index('TFs')['id'])
        interactions['TF_celltype_id'] = interactions['TF_celltype'].map(celltypes_id.set_index('celltypes')['id'])
        
    return interactions

def get_spatial_contact_matrix(adata: AnnData, id_dict, celltype='cell_type'):
    if 'x_coord' and 'y_coord' not in adata.obs.columns:
        adata.obs['x_coord'] = adata.obsm['spatial'][:, 0]
        adata.obs['y_coord'] = adata.obsm['spatial'][:, 1]
    distance_threshold = (adata.obs['x_coord'].max() - adata.obs['x_coord'].min() + adata.obs['y_coord'].max() - adata.obs['y_coord'].min()) / 8
    cells = np.array(adata.obs[['x_coord', 'y_coord']])
    dist_matrix = distance_matrix(cells, cells)
    dist_matrix = pd.DataFrame(dist_matrix, index=adata.obs.index, columns=adata.obs.index)

    celltype_map = adata.obs[celltype].to_dict()
    dist_matrix.index = dist_matrix.index.map(celltype_map)
    dist_matrix.columns = dist_matrix.columns.map(celltype_map)
    celltypes = pd.unique(adata.obs[celltype])

    close_pairs = dist_matrix < distance_threshold
    stacked_pairs = close_pairs.stack()
    grouped_sum = stacked_pairs.groupby(level=[0, 1]).sum()
    spatial_matrix = grouped_sum.unstack()
    spatial_matrix = spatial_matrix.fillna(0)
    celltype_count = adata.obs[celltype].value_counts().to_dict()

    for i in celltypes:
        for j in celltypes:
            if i == j:
                spatial_matrix.loc[i, j] = 0
            else:
                spatial_matrix.loc[i, j] = spatial_matrix.loc[i, j] / (celltype_count[i] * celltype_count[j])
    spatial_interactions = pd.DataFrame(columns=['cell0', 'cell1', 'weight'])
    for i in spatial_matrix.columns:
        for j in spatial_matrix.index:
            if i != j and i in id_dict['cell']['celltypes'].values and j in id_dict['cell']['celltypes'].values:
                spatial_interactions = pd.concat([spatial_interactions, pd.DataFrame({'cell0': i, 'cell1': j, 'weight': spatial_matrix.loc[i, j]}, index=[0])])

    spatial_interactions['cell0_id'] = spatial_interactions['cell0'].map(id_dict['cell'].set_index('celltypes')['id'])
    spatial_interactions['cell1_id'] = spatial_interactions['cell1'].map(id_dict['cell'].set_index('celltypes')['id'])

    return spatial_interactions

def create_hetero_data(interactions,  spatial_interactions, id_dict, LR_TF=False):
  
    cell2ligand = interactions[['sourcecelltype_id', 'ligand_id', 'ligand_expression']].drop_duplicates()
    cell2receptor = interactions[['targetcelltype_id', 'receptor_id', 'receptor_expression']].drop_duplicates()
    ligand2receptor = interactions[['ligand_id', 'receptor_id', 'LR_expression']].drop_duplicates()
    ligand2receptor = ligand2receptor.groupby(['ligand_id', 'receptor_id'])['LR_expression'].mean().reset_index()
    
    edge_index_dict = {}
    edge_attr_dict = {}

    edge_index_dict[('sourcecell', 'to', 'ligand')] = torch.tensor(cell2ligand[['sourcecelltype_id', 'ligand_id']].values).t().long()
    edge_index_dict[('ligand', 'to', 'sourcecell')] = torch.tensor(cell2ligand[['ligand_id', 'sourcecelltype_id']].values).t().long()
    edge_index_dict[('ligand', 'to', 'receptor')] = torch.tensor(ligand2receptor[['ligand_id', 'receptor_id']].values).t().long()
    edge_index_dict[('receptor', 'to', 'ligand')] = torch.tensor(ligand2receptor[['receptor_id', 'ligand_id']].values).t().long()
    edge_index_dict[('receptor', 'to', 'targetcell')] = torch.tensor(cell2receptor[['receptor_id', 'targetcelltype_id']].values).t().long()
    edge_index_dict[('targetcell', 'to', 'receptor')] = torch.tensor(cell2receptor[['targetcelltype_id', 'receptor_id']].values).t().long()

    edge_attr_dict[('sourcecell', 'to', 'ligand')] = torch.tensor(cell2ligand['ligand_expression'].values).float().view(-1, 1)
    edge_attr_dict[('ligand', 'to', 'sourcecell')] = torch.tensor(cell2ligand['ligand_expression'].values).float().view(-1, 1)
    edge_attr_dict[('ligand', 'to', 'receptor')] = torch.tensor(ligand2receptor['LR_expression'].values).float().view(-1, 1)
    edge_attr_dict[('receptor', 'to', 'ligand')] = torch.tensor(ligand2receptor['LR_expression'].values).float().view(-1, 1)
    edge_attr_dict[('receptor', 'to', 'targetcell')] = torch.tensor(cell2receptor['receptor_expression'].values).float().view(-1, 1)
    edge_attr_dict[('targetcell', 'to', 'receptor')] = torch.tensor(cell2receptor['receptor_expression'].values).float().view(-1, 1)
    if LR_TF:
        receptor2TF = interactions[['receptor_id', 'TF_id', 'receptor_expression']].drop_duplicates()
        edge_index_dict['receptor', 'to', 'TF'] = torch.tensor(receptor2TF[['receptor_id', 'TF_id']].values).t().long()
        edge_index_dict['TF', 'to', 'receptor'] = torch.tensor(receptor2TF[['TF_id', 'receptor_id']].values).t().long()
        edge_attr_dict['receptor', 'to', 'TF'] = torch.tensor(receptor2TF['receptor_expression'].values).float().view(-1, 1)
        edge_attr_dict['TF', 'to', 'receptor'] = torch.tensor(receptor2TF['receptor_expression'].values).float().view(-1, 1)

    if spatial_interactions is not None:
        edge_index_dict['sourcecell', 'to', 'targetcell'] = torch.tensor(spatial_interactions[['cell0_id', 'cell1_id']].values).t().long()
        edge_index_dict['targetcell', 'to', 'sourcecell'] = torch.tensor(spatial_interactions[['cell1_id', 'cell0_id']].values).t().long()

        edge_attr_dict['sourcecell', 'to', 'targetcell'] = torch.tensor(spatial_interactions['weight'].values).float().view(-1, 1)
        edge_attr_dict['targetcell', 'to', 'sourcecell'] = torch.tensor(spatial_interactions['weight'].values).float().view(-1, 1)


    hetero_data = HeteroData()
    for i in id_dict.keys():
        hetero_data[i] = {'x': torch.tensor(id_dict[i]['id'].values).long()}
    hetero_data.edge_index_dict = edge_index_dict
    hetero_data.edge_attr_dict = edge_attr_dict

    return hetero_data

# Data saving and loading
def save_data(data: any, 
              path: str):
    with gzip.open(path, 'wb') as f:
        pickle.dump(data, f)

def load_data(path: str):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

## python preprocessing.py --id lusc_st --source_dir /p300s/xiaojf_group/xuchenle/DLCCC/rctd/st/lusc_st.h5ad --data_dir /p300s/xiaojf_group/xuchenle/DLCCC/rctd/st/preprocess/1211_st/ --celltype celltype  --LR 0 --spatial 1 --TF 1

def preprocess_data(id, source_dir, data_dir, deg=True, prop=0.1, celltype='cell_type', LR=1, spatial=0, TF=0):
    """
    Process spatial transcriptomics data and create heterogeneous graph data.
    
    Parameters:
    -----------
    id : str
        Identifier for the dataset
    source_dir : str
        Path to the source h5ad file
    data_dir : str
        Directory to save processed data
    deg : bool, default=True
        Whether to use differentially expressed genes
    prop : float, default=0.1
        Proportion threshold for gene expression
    celltype : str, default='cell_type'
        Column name for cell type in the AnnData object
    LR : int, default=1
        Whether to use ligand-receptor information (1=yes, 0=no)
    spatial : int, default=0
        Whether to use spatial information (1=yes, 0=no)
    TF : int, default=0
        Whether to use transcription factor information (1=yes, 0=no)
    
    Returns:
    --------
    dict
        Dictionary containing the processed data, interactions, and ID mappings
    """
    print(f'Processing {id}......')

    save_dir = f'{data_dir}{id}/'
    make_dir(save_dir)
    
    adata = sc.read_h5ad(source_dir) # data loading
    # reference data loading
    interactions_path = '../LR_Base/LR_interactions.csv'
    complex_nodes_path = '../LR_Base/complexes_df.csv'

    LR_interactions = pd.read_csv(interactions_path)
    complex_nodes = pd.read_csv(complex_nodes_path)

    if LR:
        LR_nodes = list(set(LR_interactions['ligand'].tolist() + LR_interactions['receptor'].tolist() + complex_nodes['subunit'].tolist()))
        cell_nodes_exp = get_gene_expression(adata, LR_nodes=LR_nodes, deg=deg, prop=prop, celltype=celltype)
        complex_expr_df = calculate_complex_expr(complex_nodes, cell_nodes_exp)
        interactions = get_interactions(cell_nodes_exp, complex_expr_df, LR_interactions=LR_interactions)
        id_dict = create_id_dict(interactions, TF=TF)
        interactions = transfer_id(interactions, id_dict, TF=TF)

    if TF:
        rec2TF_path = '../LR_Base/LRTF.csv'
        rec2TF = pd.read_csv(rec2TF_path)
        LR_nodes = list(set(LR_interactions['ligand'].tolist() + LR_interactions['receptor'].tolist() + complex_nodes['subunit'].tolist() + rec2TF['TF'].tolist()))
        cell_nodes_exp = get_gene_expression(adata, LR_nodes=LR_nodes, deg=deg, prop=prop, celltype=celltype)
        complex_expr_df = calculate_complex_expr(complex_nodes, cell_nodes_exp)
        interactions = get_interactions(cell_nodes_exp, complex_expr_df, rec2TF=rec2TF)
        id_dict = create_id_dict(interactions, TF=TF)
        interactions = transfer_id(interactions, id_dict, TF=TF)

    if spatial:
        spatial_interactions = get_spatial_contact_matrix(adata, id_dict, celltype)
        data = create_hetero_data(interactions, spatial_interactions, id_dict, LR_TF=TF)
    else:
        data = create_hetero_data(interactions, spatial_interactions=None, id_dict=id_dict, LR_TF=TF)

    # Save data
    save_data(data, f'{save_dir}{id}_data.pkl')
    save_data(interactions, f'{save_dir}{id}_LR.pkl')
    save_data(id_dict, f'{save_dir}{id}_id.pkl')
    print('Data saved successfully!')
    
    return {'data': data, 'interactions': interactions, 'id_dict': id_dict}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Visium data')
    parser.add_argument('--id', type=str, help='File name')
    parser.add_argument('--source_dir', type=str, help='File directory')
    parser.add_argument('--data_dir', type=str, help='Save directory')
    parser.add_argument('--deg', type=bool, help='Whether to use DEG', default=True)
    parser.add_argument('--prop', type=float, help='Proportion of gene expression', default=0.1)
    parser.add_argument('--celltype', type=str, help='Cell type column name', default='cell_type')
    parser.add_argument('--LR', type=int, help='Whether to use LR information', default=1)
    parser.add_argument('--spatial', type=int, help='Whether to use spatial information', default=0)
    parser.add_argument('--TF', type=int, help='Whether to use TF information', default=0)

    args = parser.parse_args()
    
    preprocess_data(
        id=args.id,
        source_dir=args.source_dir, 
        data_dir=args.data_dir,
        deg=args.deg,
        prop=args.prop,
        celltype=args.celltype,
        LR=args.LR,
        spatial=args.spatial,
        TF=args.TF
    )