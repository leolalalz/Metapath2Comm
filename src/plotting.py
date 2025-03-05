import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sns
from d3blocks import D3Blocks
from typing import Dict, List

def get_plot_data(id_dict: dict, 
                  result_df: pd.DataFrame, 
                  id: bool=True):
    celltype_id = id_dict['celltype']
    grouped_expression = result_df.groupby('label')['score'].sum().reset_index()
    result = []
    for i in result_df['label'].unique():
        sub_result = []
        #sub_result.append(i)
        sub_result.append(i.split('_')[0])
        sub_result.append(i.split('_')[-1])
        sub_result.append(grouped_expression.loc[grouped_expression['label'] == i]['score'].values[0])
        sub_result.append(result_df.loc[result_df['label'] == i].shape[0])
        result.append(sub_result)
    result = pd.DataFrame(result, columns=['source', 'target', 'weight', 'count']).sort_values(by='weight', ascending=False).reset_index(drop=True)
    if id:
        celltype_id = celltype_id.astype(str)
        result = pd.merge(result, celltype_id, left_on='source', right_on='id', how='left')
        result['source'] = result['celltype']
        result = result.drop(columns=['id', 'celltype'])

        result = pd.merge(result, celltype_id, left_on='target', right_on='id', how='left')
        result['target'] = result['celltype']
        result = result.drop(columns=['id', 'celltype'])
    return result


def get_relation_matrix(
    result: pd.DataFrame, 
    types: list
) -> pd.DataFrame:

    relation_matrix = pd.DataFrame(np.zeros((len(types), len(types))), index=types, columns=types)
    for _, row in result.iterrows():
        relation_matrix.loc[row['source'], row['target']] = row['weight']

    return relation_matrix

def generate_random_color() -> str:
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def network_plot(
    result: pd.DataFrame,
    title: str = 'Network Plot',
    save: bool = False,
    filepath: str = './network_plot.png'
) -> None:
    palette = {cell_type: generate_random_color() for cell_type in result['source'].unique()}
    sr_network_graph = nx.DiGraph()
    sr_network_graph.add_nodes_from(result['source'].unique())
    plt.figure(figsize=(8, 8))  # 增加图的尺寸
    plt.tight_layout()

    for _, row in result.iterrows():
        sr_network_graph.add_weighted_edges_from(
                    [(row[0], row[1], row[2])])

    node_color = [palette[i] for i in result['source'].unique()]
    edge_color = [palette[i[0]] for i in sr_network_graph.edges]
    pos = nx.circular_layout(sr_network_graph)  
    nx.draw(sr_network_graph, 
            pos=pos,
            linewidths=2.0, edgecolors='black',
            node_color=node_color,
            node_size=1000,
            connectionstyle='arc3,rad=0.15', arrowstyle='-|>',
            edge_color=edge_color,
            width=[float(v['weight']/100) for (r, c, v) in sr_network_graph.edges(data=True)])

    labels = nx.draw_networkx_labels(sr_network_graph, pos={node: (x * 1.1, y * 1.1) for node, (x, y) in pos.items()},
                                    font_size=12, font_color='black')

    plt.title(title, fontsize=16)  # 添加标题
    plt.tight_layout()

    if save:
        plt.savefig(filepath, dpi=300)
    plt.show()
    plt.close()

from typing import List, Tuple
def heatmap(
    relation_matrix: pd.DataFrame, 
    title: str, 
    cmap: str = 'Blues', 
    size: Tuple[int, int] = (10, 8), 
    hist: bool = False,
    txt: bool = False, 
    save: bool = False, 
    filepath: str = './heatmap.png', 
    show: bool = True, 
    rect: List[float] = [0, 0, 0, 1]
) -> None:
    source_cell = relation_matrix.index.tolist()    
    target_cell = relation_matrix.columns.tolist()

    relation_matrix = np.array(relation_matrix.values, dtype=float)

    # Create the layout for the main plot and histograms
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(2, 3, width_ratios=(8, 1, 0.2), height_ratios=(1, 8), wspace=0.0, hspace=0.0)

    # Create heatmap
    ax_heatmap = fig.add_subplot(gs[1, 0])
    im = ax_heatmap.imshow(relation_matrix, cmap=cmap, aspect='auto', interpolation='none')
    ax_heatmap.grid(False)

    if hist:
        # Create row histogram
        ax_row_hist = fig.add_subplot(gs[1, 1], sharey=ax_heatmap)
        row_sums = relation_matrix.sum(axis=1)
        sns.barplot(y=source_cell, x=row_sums, ax=ax_row_hist, palette=['#ADD8E6'])
        ax_row_hist.set_ylabel('')
        ax_row_hist.set_xlabel('Sum')
        plt.setp(ax_row_hist.get_yticklabels(), visible=False)
        for i, v in enumerate(row_sums):
            ax_row_hist.text(0.9, i, f"{v:.2f}", color='white' if im.cmap(im.norm(v))[0] > 1 else 'black', va='center')

        # Create column histogram
        ax_col_hist = fig.add_subplot(gs[0, 0], sharex=ax_heatmap)
        col_sums = relation_matrix.sum(axis=0)
        sns.barplot(x=target_cell, y=col_sums, ax=ax_col_hist, palette=['#ADD8E6'])
        ax_col_hist.set_xlabel('')
        ax_col_hist.set_ylabel('Sum')
        plt.setp(ax_col_hist.get_xticklabels(), visible=False)
        for i, v in enumerate(col_sums):
            ax_col_hist.text(i, 0.9, f"{v:.2f}", color='white' if im.cmap(im.norm(v))[0] > 1 else 'black', ha='center')

    for spine in ax_heatmap.spines.values():
        spine.set_visible(True)
        
    # Set ticks and labels
    ax_heatmap.set_xticks(np.arange(len(target_cell)))    
    ax_heatmap.set_yticks(np.arange(len(source_cell)))    
    ax_heatmap.set_xticklabels(target_cell)        
    ax_heatmap.set_yticklabels(source_cell) 

    plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text to the heatmap
    if txt:
        for i in range(len(source_cell)):
            for j in range(len(target_cell)):
                formatted_number = f"{relation_matrix[i, j]:.2f}"
                # 获取当前单元格的颜色值
                color_value = im.cmap(im.norm(relation_matrix[i, j]))
                # 计算颜色的亮度
                brightness = (0.299 * color_value[0] + 0.587 * color_value[1] + 0.114 * color_value[2])
                # 根据亮度选择文字颜色
                text_color = 'black' if brightness > 0.5 else 'white'
                ax_heatmap.text(j, i, formatted_number, ha="center", va="center", color=text_color)

    # Create a separate axis for the color bar
    ax_cbar = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(im, cax=ax_cbar, orientation='vertical')

    
    fig.suptitle(title, fontsize=20, fontweight='bold')   

    fig.tight_layout(rect=rect) 
    if save:
        plt.savefig(filepath, dpi=300)
    if show:
        plt.show() 
    plt.close()

#heatmap(relation_matrix, 'Communication Matrix', cmap='Blues', size=(10, 10), txt=True, hist=True, save=False, show=True, rect=[0, 1, 1, 1])


def chord_diagram(
    result: pd.DataFrame, 
    filepath: str, 
    title: str
) -> None:
    d3 = D3Blocks()
    d3.chord(result, filepath=filepath, cmap='tab20c', title=title, fontsize=12, figsize=[1000, 1000])


# modified version
import random
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from d3blocks import D3Blocks


def get_plot_data(id_dict: dict, result_df: pd.DataFrame, use_id: bool = True) -> pd.DataFrame:
    """
    Prepare plot data by splitting the 'label' column into source and target,
    summing scores, and optionally mapping node ids.
    """
    celltype_id = id_dict.get('celltype')
    grouped = result_df.groupby('label')['score'].sum().reset_index()
    
    records = []
    for label in result_df['label'].unique():
        source = label.split('_')[0]
        target = label.split('_')[-1]
        weight = grouped.loc[grouped['label'] == label, 'score'].values[0]
        count = result_df.loc[result_df['label'] == label].shape[0]
        records.append([source, target, weight, count])
        
    df = pd.DataFrame(records, columns=['source', 'target', 'weight', 'count'])
    df.sort_values(by='weight', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if use_id and celltype_id is not None:
        celltype_id = celltype_id.astype(str)
        df = df.merge(celltype_id, left_on='source', right_on='id', how='left')
        df['source'] = df['celltype']
        df.drop(columns=['id', 'celltype'], inplace=True)
        df = df.merge(celltype_id, left_on='target', right_on='id', how='left')
        df['target'] = df['celltype']
        df.drop(columns=['id', 'celltype'], inplace=True)
    return df


def get_relation_matrix(result: pd.DataFrame, types: list) -> pd.DataFrame:
    """
    Generate a relation matrix of given types using the 'weight' for each source-target pair.
    """
    matrix = pd.DataFrame(0, index=types, columns=types, dtype=float)
    for _, row in result.iterrows():
        matrix.loc[row['source'], row['target']] = row['weight']
    return matrix


def generate_random_color() -> str:
    """Generate a random hex color string."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def network_plot(result: pd.DataFrame,
                 title: str = 'Network Plot',
                 save: bool = False,
                 filepath: str = './network_plot.png') -> None:
    """
    Plot a directed network graph using the provided result dataframe.
    """
    unique_nodes = result['source'].unique()
    palette = {node: generate_random_color() for node in unique_nodes}
    
    # Build graph
    G = nx.DiGraph()
    G.add_nodes_from(unique_nodes)
    for _, row in result.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
    
    pos = nx.circular_layout(G)
    node_colors = [palette[node] for node in G.nodes()]
    edge_colors = [palette[u] for u, v in G.edges()]
    widths = [G[u][v]['weight'] / 100.0 for u, v in G.edges()]

    plt.figure(figsize=(8, 8))
    nx.draw(G, pos=pos, with_labels=True,
            node_color=node_colors, edge_color=edge_colors,
            width=widths, node_size=1000, arrowstyle='-|>',
            connectionstyle='arc3,rad=0.15', linewidths=2.0,
            edgecolors='black')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(filepath, dpi=300)
    plt.show()
    plt.close()


def heatmap(relation_matrix: pd.DataFrame,
            title: str,
            cmap: str = 'Blues',
            size: Tuple[int, int] = (10, 8),
            show_hist: bool = False,
            show_text: bool = False,
            save: bool = False,
            filepath: str = './heatmap.png',
            show: bool = True,
            rect: List[float] = [0, 0, 0, 1]) -> None:
    """
    Draw a heatmap with the relation matrix and optionally display histograms and text annotations.
    """
    source_labels = relation_matrix.index.tolist()
    target_labels = relation_matrix.columns.tolist()
    matrix = relation_matrix.values.astype(float)
    
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(2, 3,
                          width_ratios=(8, 1, 0.2),
                          height_ratios=(1, 8),
                          wspace=0.0, hspace=0.0)
    
    ax_heat = fig.add_subplot(gs[1, 0])
    im = ax_heat.imshow(matrix, cmap=cmap, aspect='auto', interpolation='none')
    ax_heat.set_xticks(np.arange(len(target_labels)))
    ax_heat.set_yticks(np.arange(len(source_labels)))
    ax_heat.set_xticklabels(target_labels, rotation=45, ha="right")
    ax_heat.set_yticklabels(source_labels)
    for spine in ax_heat.spines.values():
        spine.set_visible(True)
        
    if show_text:
        for i in range(len(source_labels)):
            for j in range(len(target_labels)):
                val = matrix[i, j]
                rgba = im.cmap(im.norm(val))
                brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = 'black' if brightness > 0.5 else 'white'
                ax_heat.text(j, i, f"{val:.2f}", ha='center', va='center', color=text_color)
    
    if show_hist:
        ax_row = fig.add_subplot(gs[1, 1], sharey=ax_heat)
        row_sums = matrix.sum(axis=1)
        sns.barplot(x=row_sums, y=source_labels, ax=ax_row, palette=['#ADD8E6'])
        ax_row.set_xlabel('Sum')
        ax_row.tick_params(labelleft=False)
        for i, v in enumerate(row_sums):
            ax_row.text(v, i, f"{v:.2f}", va='center', color='black')
        
        ax_col = fig.add_subplot(gs[0, 0], sharex=ax_heat)
        col_sums = matrix.sum(axis=0)
        sns.barplot(x=target_labels, y=col_sums, ax=ax_col, palette=['#ADD8E6'])
        ax_col.set_ylabel('Sum')
        ax_col.tick_params(labelbottom=False)
        for i, v in enumerate(col_sums):
            ax_col.text(i, v, f"{v:.2f}", ha='center', color='black')
    
    ax_cbar = fig.add_subplot(gs[:, 2])
    fig.colorbar(im, cax=ax_cbar, orientation='vertical')
    
    fig.suptitle(title, fontsize=20, fontweight='bold')
    fig.tight_layout(rect=rect)
    if save:
        plt.savefig(filepath, dpi=300)
    if show:
        plt.show()
    plt.close()


def chord_diagram(result: pd.DataFrame, filepath: str, title: str) -> None:
    """
    Generate a chord diagram using D3Blocks.
    """
    d3 = D3Blocks()
    d3.chord(result, filepath=filepath, cmap='tab20c', title=title, fontsize=12, figsize=[1000, 1000])