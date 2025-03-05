import numpy as np
import pandas as pd
import torch
import os
import pickle
import gzip
import warnings
warnings.filterwarnings('ignore')
import argparse
from preprocessing import make_dir, load_data
from sklearn.preprocessing import MinMaxScaler


def cosine_similarity(embedding_dict: dict, 
                      edge_index_dict: dict):
    """计算嵌入向量之间的余弦相似度"""
    similarity_dict = {}
    for i in edge_index_dict.keys():
        source, _, target = i
        if source in embedding_dict.keys() and target in embedding_dict.keys():
            dot_products = np.dot(embedding_dict[source], embedding_dict[target].T)
            norms_source = np.linalg.norm(embedding_dict[source], axis=1)
            norms_target = np.linalg.norm(embedding_dict[target], axis=1)
            norms_product = np.outer(norms_source, norms_target)
            similarity = dot_products / norms_product
            similarity_dict[i] = similarity
    return similarity_dict


def get_embedding_dict(model, key_list):
    """从模型中获取不同实体类型的嵌入向量字典"""
    embedding_dict = {}
    for key in key_list:
        embedding_dict[key] = model(key).to('cpu').detach().numpy()
    return embedding_dict


def process_similarity(similarity_dict, entity_pair, cutoff, columns):
    """处理相似度矩阵，提取高于阈值的相似度对"""
    array = similarity_dict[entity_pair]
    df = pd.DataFrame(np.vstack((np.where(array > cutoff), array[np.where(array > cutoff)])).T, columns=columns)
    df = df.astype({col: int for col in columns[:-1]})
    df = df.astype({col: str for col in columns[:-1]})
    return df


def get_communication_result(model: torch.nn.Module, 
                             data, 
                             interactions: pd.DataFrame, 
                             TF=0, 
                             spatial=0,
                             cutoff: float=0):
    """
    从训练好的模型中获取细胞通信结果
    
    参数:
        model: 训练好的MetaPath2Vec模型
        data: 异构图数据
        interactions: 交互数据
        TF: 是否包含转录因子信息
        spatial: 是否包含空间信息
        cutoff: 相似度截断阈值
    
    返回:
        DataFrame: 细胞通信结果表
    """
    scaler = MinMaxScaler()
    embedding_dict = get_embedding_dict(model, ['ligand', 'receptor', 'sourcecell', 'targetcell'])
    similarity_dict = cosine_similarity(embedding_dict, data.edge_index_dict)

    # 处理配体-受体相似度
    lr_df = process_similarity(similarity_dict, ('ligand', 'to', 'receptor'), cutoff, ['ligand', 'receptor', 'similarity'])
    lr_df['l2r'] = lr_df['ligand'] + '_' + lr_df['receptor']

    # 处理细胞-配体相似度
    cl_df = process_similarity(similarity_dict, ('sourcecell', 'to', 'ligand'), cutoff, ['sourcecell', 'ligand', 'similarity'])
    cl_df['c2l'] = cl_df['sourcecell'] + '_' + cl_df['ligand']

    # 处理细胞-受体相似度
    cr_df = process_similarity(similarity_dict, ('targetcell', 'to', 'receptor'), cutoff, ['targetcell', 'receptor', 'similarity'])
    cr_df['c2r'] = cr_df['targetcell'] + '_' + cr_df['receptor']

    # 合并相似度信息到交互数据
    interactions['l2r'] = interactions['ligand_id'].astype(str) + '_' + interactions['receptor_id'].astype(str)
    interactions['c2l'] = interactions['sourcecelltype_id'].astype(str) + '_' + interactions['ligand_id'].astype(str)
    interactions['c2r'] = interactions['targetcelltype_id'].astype(str) + '_' + interactions['receptor_id'].astype(str)
    
    # 筛选匹配的交互
    result_df = interactions.loc[(interactions['c2l'].isin(cl_df['c2l'])) & 
                                (interactions['c2r'].isin(cr_df['c2r'])) & 
                                (interactions['l2r'].isin(lr_df['l2r']))]
    
    # 添加相似度信息
    result_df['l2r_sim'] = result_df['l2r'].map(lr_df.set_index('l2r')['similarity'])
    result_df['c2l_sim'] = result_df['c2l'].map(cl_df.set_index('c2l')['similarity'])
    result_df['c2r_sim'] = result_df['c2r'].map(cr_df.set_index('c2r')['similarity'])
    result_df = result_df.astype({'l2r_sim': float, 'c2l_sim': float, 'c2r_sim': float})
    
    # 归一化相似度
    result_df[['c2l_sim_norm', 'c2r_sim_norm', 'l2r_sim_norm']] = scaler.fit_transform(result_df[['c2l_sim', 'c2r_sim', 'l2r_sim']])
    
    # 基于表达水平进行过滤
    result_df = result_df.loc[(result_df['ligand_expression'] > np.percentile(result_df['ligand_expression'].dropna(), 10)) & 
                              (result_df['receptor_expression'] > np.percentile(result_df['receptor_expression'].dropna(), 10))]
    result_df = result_df.loc[(result_df['c2l_sim_norm'] != 0) & 
                              (result_df['c2r_sim_norm'] != 0) & 
                              (result_df['l2r_sim_norm'] != 0)]

    # 处理转录因子相关数据（如果包含）
    if TF:
        embedding_dict['TF'] = model('TF').to('cpu').detach().numpy()
        similarity_dict = cosine_similarity(embedding_dict, data.edge_index_dict)
        rTF_df = process_similarity(similarity_dict, ('receptor', 'to', 'TF'), cutoff, ['receptor', 'TF', 'similarity'])
        rTF_df['r2TF'] = rTF_df['receptor'] + '_' + rTF_df['TF']
        result_df['r2TF'] = result_df['receptor_id'].astype(str) + '_' + result_df['TF_id'].astype(str)
        result_df = result_df.loc[result_df['r2TF'].isin(rTF_df['r2TF'])]
        result_df['r2TF_sim'] = result_df['r2TF'].map(rTF_df.set_index('r2TF')['similarity'])
        result_df = result_df.astype({'r2TF_sim': float})
        result_df[['r2TF_sim_norm']] = scaler.fit_transform(result_df[['r2TF_sim']])
        result_df = result_df.loc[result_df['r2TF_sim_norm'] != 0]
        
        # 计算总分
        result_df['score'] = (result_df['c2l_sim_norm'] + result_df['l2r_sim_norm'] + 
                             result_df['c2r_sim_norm'] + result_df['r2TF_sim_norm']) * \
                             result_df['LR_expression'] * (result_df['ligand_props'] + result_df['receptor_props'])
        
        # 生成最终结果表
        result_df_final = result_df[['ligand', 'receptor', 'TF', 'ligand_celltype', 'receptor_celltype', 'TF_celltype', 
                'ligand_expression', 'receptor_expression', 'TF_expression', 'ligand_props', 'receptor_props', 'TF_props', 
                'l2r_sim', 'c2l_sim', 'c2r_sim', 'r2TF_sim', 'score']]
    else:
        # 不包含转录因子信息时的计分
        result_df['score'] = (result_df['c2l_sim_norm'] + result_df['l2r_sim_norm'] + 
                             result_df['c2r_sim_norm']) * result_df['LR_expression'] * \
                             (result_df['ligand_props'] + result_df['receptor_props'])
        
        # 生成最终结果表
        result_df_final = result_df[['ligand', 'receptor', 'ligand_celltype', 'receptor_celltype', 
                'ligand_expression', 'receptor_expression', 'ligand_props', 'receptor_props', 
                'l2r_sim', 'c2l_sim', 'c2r_sim', 'score']]
    
    # 过滤同一细胞类型的通信
    result_df_final = result_df_final.loc[result_df_final['ligand_celltype'] != result_df_final['receptor_celltype']]
    
    # 添加标识信息
    result_df_final['label'] = result_df_final['ligand_celltype'] + '->' + result_df_final['receptor_celltype']
    result_df_final['LR'] = result_df_final['ligand'] + '->' + result_df_final['receptor']
    result_df_final['identifier'] = result_df_final['label'] + '_' + result_df_final['LR']
    
    # 按分数排序
    result_df_final = result_df_final.sort_values(by='score', ascending=False).reset_index(drop=True)

    return result_df_final


def generate_results(id, model_dir, data_dir, result_dir, train_count="0", TF=0, cutoff=0):
    """
    生成并保存细胞通信结果
    
    参数:
        id: 数据集标识
        model_dir: 模型目录
        data_dir: 数据目录
        result_dir: 结果保存目录
        train_count: 训练标识
        TF: 是否包含转录因子信息
        cutoff: 相似度截断阈值
    
    返回:
        DataFrame: 细胞通信结果表
    """
    # 确保结果目录存在
    save_path = os.path.join(result_dir, id)
    make_dir(save_path)
    
    # 加载模型和数据
    model_file = os.path.join(model_dir, f"{id}", f"{id}_{train_count}_cpu.pkl")
    data_file = os.path.join(data_dir, f"{id}", f"{id}_data.pkl")
    interactions_file = os.path.join(data_dir, f"{id}", f"{id}_LR.pkl")
    
    model = torch.load(model_file)
    data = load_data(data_file)
    interactions = load_data(interactions_file)

    # 获取通信结果
    result_df = get_communication_result(
        model=model, 
        data=data, 
        interactions=interactions, 
        TF=TF,
        cutoff=cutoff
    )
    
    # 保存结果
    output_file = os.path.join(save_path, f"{id}_{train_count}_result.csv")
    result_df.to_csv(output_file, index=False)
    print(f'结果已保存至: {output_file}')
    
    return result_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成细胞通信结果')
    parser.add_argument('--id', type=str, help='数据集标识')
    parser.add_argument('--model_dir', type=str, help='模型目录')
    parser.add_argument('--data_dir', type=str, help='数据目录')
    parser.add_argument('--result_dir', type=str, help='结果保存目录')
    parser.add_argument('--train_count', type=str, default='0', help='训练标识')
    parser.add_argument('--TF', type=int, default=0, help='是否包含转录因子信息 (0=否, 1=是)')
    parser.add_argument('--cutoff', type=float, default=0, help='相似度截断阈值')

    args = parser.parse_args()
    
    generate_results(
        id=args.id,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        train_count=args.train_count,
        TF=args.TF,
        cutoff=args.cutoff
    )