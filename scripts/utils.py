import torch
import numpy as np
import pickle
import scipy.sparse as sp
import os
from tabulate import tabulate
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from scripts.adjacent_matrix_norm import calculate_scaled_laplacian, calculate_symmetric_normalized_laplacian, calculate_symmetric_message_passing_adj, calculate_transition_matrix, calculate_random_walk_matrix


def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]

def load_raw_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return adj_mx

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)  # 将邻接矩阵稀疏化, 只将非零数据以列表形式保存 如(206, 155)	0.9271291   (206, 159)	0.78106517
    rowsum = np.array(adj.sum(1)).flatten()  # 计算每一行的和 rowsum 形状 [1,207]
    d_inv = np.power(rowsum, -1).flatten()  # 对rowsum取倒数,数值变成0-1之间,越小说明关注度越高
    d_inv[np.isinf(d_inv)] = 0.  # 把无穷小转化为0,d_inv表示每个传感器重要度倒数的列表
    d_mat = sp.diags(d_inv)  # 使用d_inv初始化对角矩阵d_mat
    return d_mat.dot(adj).astype(np.float32).todense()  # 矩阵乘法 d_mat * adj

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std,device):
        self.mean = mean
        self.std = std
        self.device = device

    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * torch.tensor(self.std).to(device=self.device)) + torch.tensor(self.mean).to(device=self.device)
        #return (data * self.std) + self.mean

def create_save_path(base_path,model_name,train_type):
    model_folder = os.path.join(base_path, train_type,model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)  # 如果文件夹不存在则创建
    print(f"Directory created successfully at: {model_folder}")
    return model_folder

def save_log_file(log_file_path,content):
    f = open(log_file_path, 'w')
    f.write(content + '\n')
    return f


def print_args(args, log_file_path):
    # 如果是 Namespace，转成 dict；如果已经是 dict 就直接用
    if not isinstance(args, dict):
        args_dict = vars(args)
    else:
        args_dict = args

    # 把参数变成列表方便 tabulate
    args_list = [(key, value) for key, value in args_dict.items()]

    # 格式化打印
    print(tabulate(args_list, headers=["Argument", "Value"], tablefmt="pretty"))
    format_args = tabulate(args_list, headers=["Argument", "Value"], tablefmt="pretty")

    # 保存到日志文件
    f = save_log_file(log_file_path, format_args)
    return f

#恢复shuffle的索引
def restore_order(shuffled_indices,predictions):
    restored_indices = torch.argsort(torch.tensor(shuffled_indices))
    return predictions[restored_indices]

def save_experiment_result(model_name,STmodel_name,feature_name, feature_data, base_path,train_type):
    if train_type == 'STKri':
        if model_name != 'GT':
            dataset_folder = os.path.join(base_path,train_type, f'{model_name}_{STmodel_name}')
        else:
            dataset_folder = os.path.join(base_path, train_type, r'GT')
    else:
        dataset_folder = os.path.join(base_path, train_type, model_name)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)  # 如果文件夹不存在则创建
    feature_file = os.path.join(dataset_folder, f'{feature_name}_prediction.npy')
    np.save(feature_file, feature_data)
    print(f"Experiment Results Saved to {feature_file}")

def save_param_experiment_result(model_name,feature_name, feature_data, base_path,k_param,k,layer_param,num_layer,d_param,d_model):
    dataset_folder = os.path.join(base_path, model_name)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)  # 如果文件夹不存在则创建
    if k_param==True:
        feature_dir=os.path.join(dataset_folder,f'k_{k}')
        feature_file = os.path.join(feature_dir, f'{feature_name}_prediction_k_{k}.npy')
    if layer_param==True:
        feature_dir=os.path.join(dataset_folder,f'layer_{num_layer}')
        feature_file = os.path.join(feature_dir, f'{feature_name}_prediction_layer_{num_layer}.npy')
    if d_param==True:
        feature_dir=os.path.join(dataset_folder,f'd_{d_model}')
        feature_file = os.path.join(feature_dir, f'{feature_name}_prediction_d_{d_model}.npy')
    os.makedirs(feature_dir, exist_ok=True)
    np.save(feature_file, feature_data)
    print(f"Experiment Results Saved to {feature_file}")

def find_k_nearest_neighbors(era_data, cobs, cera,N,k=1):

        """
        找到每个 obs_his 站点 (N) 的 k 个近邻的 era_his 和 pan_fut 数据点。
            era_his: ndarray, (B, C, lat, lon, L) 的 ERA 历史数据
            cobs: ndarray, (N, 2) 的站点坐标 (纬度, 经度)
            cera: ndarray, (lat, lon, 2) 的 ERA 网格坐标 (纬度, 经度)
            k: int, 要找到的近邻数量

        返回:
            era_k: ndarray, (B, C, N, k, L) 的 ERA 近邻数据
            pan_k: ndarray, (B, C, N, k, L) 的 PAN 近邻数据
        """
        #将era5展平
        B,C,_,_,L=era_data.shape
        era_his=era_data.reshape(B,C,-1,L)
        # cera 和 cpan
        cera_flat = cera.reshape(-1, 2)  # (lat * lon, 2)

        # 初始化最近邻模型
        nbrs_era = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(cera_flat)

        era_k=[]
        for n in range(N):
            # 获取当前 obs_his 站点的坐标
            station_coord = np.array(cobs[n]).reshape(1,2)  # (2,)

            # 获取该站点最近的 k 个 ERA 和 PAN 网格点索引
            _, indices_era = nbrs_era.kneighbors(station_coord)
            era_his_n=era_his[:,:,indices_era,:]#era_his:(B,C,1,k,L)
            era_k.append(era_his_n)
        era_k=np.concatenate(era_k,axis=2)#era_k:(B,C,N,1,L)
        return era_k

def adj_transform(adj_mx, adj_type: str):
    """load adjacency matrix.

    Args:
        file_path (str): file path
        adj_type (str): adjacency matrix type

    Returns:
        list of numpy.matrix: list of preproceesed adjacency matrices
        np.ndarray: raw adjacency matrix
    """
    #print(os.path.abspath(file_path))
    #print(adj_type)
    #adj_mx = adj_node_index(adj_mx,adj_index)
    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [calculate_symmetric_message_passing_adj(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [calculate_transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [calculate_transition_matrix(adj_mx).T, calculate_transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == 'random_walk':
        adj = [calculate_random_walk_matrix(adj_mx)]
    elif adj_type == "original":
        adj = [adj_mx]
    elif adj_type == "asym_adj":
        adj= [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    else:
        error = 0
        assert error, "adj type not defined"
    print(adj_type, np.sum(adj), np.sum(adj_mx),'kkkkkkk')
    return adj, adj_mx

def adj_node_index(adj,index):
    """ Reorder the elements within the adjacency matrix according to the given new indexes  """
    adp = adj[index,:]
    adp2 = adp[:,index]
    return adp2

def adj_mask_unknown_node(adj, unknown_idx):
    if len(adj.shape) == 2:
        unknown_idx_adj = torch.LongTensor(unknown_idx).repeat(adj.shape[0],1)
        adj = adj.scatter(1,unknown_idx_adj,0)
        return adj
    elif len(adj.shape) == 3:
        l = unknown_idx.shape[1]
        unknown_idx_adj = torch.LongTensor(unknown_idx).repeat(1,adj.shape[1]).reshape(adj.shape[0],adj.shape[1],l)
        adj = adj.scatter(2, unknown_idx_adj, 0)
        return adj
    else:
        return adj

def load_station_splits(filename="station_split.txt"):
    train_ids, val_ids, test_ids = [], [], []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Train:"):
                train_ids = list(map(int, line.replace("Train:", "").strip().split()))
            elif line.startswith("Validation:"):
                val_ids = list(map(int, line.replace("Validation:", "").strip().split()))
            elif line.startswith("Test:"):
                test_ids = list(map(int, line.replace("Test:", "").strip().split()))
    return train_ids, val_ids, test_ids

