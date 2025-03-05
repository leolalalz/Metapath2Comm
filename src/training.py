import psutil
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import argparse
import pandas as pd
import time
from tqdm import tqdm
from copy import deepcopy
from preprocessing import make_dir, load_data
from metapath2vec import MetaPath2Vec


def train_model(id, preprocess_dir, model_path, spatial=0, TF=0, train_count="0", 
                train_epoch=400, device="cuda:0", use_tensorboard=True):
    """
    Train a MetaPath2Vec model on the given data.
    
    Parameters:
    -----------
    id : str
        Identifier for the dataset and model
    preprocess_dir : str
        The preprocessed data directory
    model_path : str
        Directory to save model and training logs
    spatial : int, default=0
        Whether to use spatial information (1=yes, 0=no)
    TF : int, default=0
        Whether to use transcription factor information (1=yes, 0=no)
    train_count : str, default="0"
        Training run identifier
    train_epoch : int, default=400
        Maximum number of training epochs
    device : str, default="cuda:0"
        Device to run training on (e.g., 'cuda:0', 'cpu')
    use_tensorboard : bool, default=True
        Whether to use TensorBoard for logging
        
    Returns:
    --------
    dict
        Dictionary containing training results and the trained model
    """
    device = device if torch.cuda.is_available() else 'cpu'
    data = load_data(f'{preprocess_dir}{id}/{id}_data.pkl')
    # Metapath setup
    if spatial and TF:
        metapath = [
            ('sourcecell', 'to', 'targetcell'), ('targetcell', 'to', 'receptor'),
            ('receptor', 'to', 'TF'), ('TF', 'to', 'receptor'),
            ('receptor', 'to', 'ligand'), ('ligand', 'to', 'sourcecell'), 
            ('sourcecell', 'to', 'ligand'), ('ligand', 'to', 'receptor'),
            ('receptor', 'to', 'TF'), ('TF', 'to', 'receptor'),
            ('receptor', 'to', 'targetcell'), ('targetcell', 'to', 'sourcecell')
        ]
    elif spatial and not TF:
        metapath = [
            ('sourcecell', 'to', 'targetcell'), ('targetcell', 'to', 'receptor'),
            ('receptor', 'to', 'ligand'), ('ligand', 'to', 'sourcecell'), 
            ('sourcecell', 'to', 'ligand'), ('ligand', 'to', 'receptor'),
            ('receptor', 'to', 'targetcell'), ('targetcell', 'to', 'sourcecell')
        ]
    elif TF and not spatial:
        metapath = [
            ('sourcecell', 'to', 'ligand'), ('ligand', 'to', 'receptor'),
            ('receptor', 'to', 'TF'), ('TF', 'to', 'receptor'),
            ('receptor', 'to', 'targetcell'), ('targetcell', 'to', 'receptor'),
            ('receptor', 'to', 'TF'), ('TF', 'to', 'receptor'),
            ('receptor', 'to', 'ligand'), ('ligand', 'to', 'sourcecell')
        ]
    else:
        metapath = [
            ('sourcecell', 'to', 'ligand'), ('ligand', 'to', 'receptor'),
            ('receptor', 'to', 'targetcell'), ('targetcell', 'to', 'receptor'),
            ('receptor', 'to', 'ligand'), ('ligand', 'to', 'sourcecell')
        ]
    
    print(f'Device: {device}')
    print(f'Metapath: {metapath}')
    
    # Initialize the model, loader, and optimizer
    model = MetaPath2Vec(
        data.edge_index_dict, embedding_dim=64, metapath=metapath,
        walk_length=len(metapath) * 6, context_size=15, walks_per_node=100,
        num_negative_samples=15, edge_attr_dict=data.edge_attr_dict, sparse=True
    ).to(device)
    
    loader = model.loader(batch_size=8, shuffle=True, num_workers=10)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    
    # TensorBoard writer initialization
    writer = None
    if use_tensorboard:
        writer = SummaryWriter(log_dir=f'{model_path}{id}_{train_count}_tensorboard_logs')
    
    # Training loop with early stopping
    record = []
    best_loss = float('inf')
    best_model = None
    early_stopping_counter = 0
    start_time = time.time()
    
    def train_epoch():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    print('Starting training...')
    for epoch in tqdm(range(1, train_epoch + 1), desc='Epoch'):
        loss = train_epoch()
        
        if use_tensorboard:
            writer.add_scalar('Loss/train', loss, epoch)  # Log loss to TensorBoard
        
        if loss < best_loss:
            best_loss = loss
            best_model = deepcopy(model)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= 20:
            print("Early stopping triggered.")
            break
        
        if epoch % 5 == 0:
            memory_info = psutil.virtual_memory()
            memory_used_gb = memory_info.used / (1024 ** 3)
            record.append([epoch, best_loss, time.time() - start_time, memory_used_gb])
        
        scheduler.step()
    
    # Training summary
    end_time = time.time()
    train_time = end_time - start_time
    print(f'Training completed in {train_time:.2f} seconds')
    print(f'Best Loss: {best_loss:.4f}')
    
    # Save results and model
    record_df = pd.DataFrame(record, columns=['epoch', 'loss', 'time', 'memory_used_gb'])
    record_df.to_csv(f'{model_path}{id}_record_{train_count}.csv', index=False)
    
    torch.save(best_model.cpu(), f'{model_path}{id}_{train_count}_cpu.pkl')
    print('Model saved successfully.')
    
    # Close TensorBoard writer
    if use_tensorboard and writer:
        writer.close()
    
    return {
        'model': best_model,
        'loss': best_loss,
        'training_time': train_time,
        'record': record_df
    }


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train MetaPath2Vec model')
    parser.add_argument('--id', type=str, help='File name')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--model_dir', type=str, help='Model directory')
    parser.add_argument('--spatial', type=int, default=0, help='Whether to use spatial data')
    parser.add_argument('--TF', type=int, default=0, help='Whether to use TF information')
    parser.add_argument('--train_count', type=str, help='Train count')
    parser.add_argument('--train_epoch', type=int, default=400, help='Train epoch')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--no_tensorboard', action='store_true', default=False, help='Disable TensorBoard logging')
    args = parser.parse_args()
    
    # Path and device setup
    
    train_model(
        id=args.id,
        preprocess_dir=args.data_dir,
        model_path=args.model_path,
        spatial=args.spatial,
        TF=args.TF,
        train_count=args.train_count,
        train_epoch=args.train_epoch,
        device=args.device,
        use_tensorboard=not args.no_tensorboard
    )