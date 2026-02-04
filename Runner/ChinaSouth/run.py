import warnings
import argparse
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from tqdm import tqdm
from data_provider.load_dataset import load_dataset
from scripts.utils import load_adj,create_save_path,print_args,save_experiment_result
from Trainer.trainer import Trainer
from scripts.metric import *
import numpy as np
import gc
import torch
import math

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings('ignore')

torch.cuda.empty_cache()
gc.collect()
parser=argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../../datasets/ChinaSouth', help='Input Data Path')
parser.add_argument('--cpan_path', type=str, default='pan_lat_lon.npy', help='Pangu Latitude and Longitude File')
parser.add_argument('--csta_path', type=str, default='obs_lat_lon.npy', help='Observation station Latitude and Longitude File')
parser.add_argument('--csta_train_path', type=str, default='obs_train_lat_lon.npy', help='Observation station Latitude and Longitude File')
parser.add_argument('--csta_val_path', type=str, default='obs_val_lat_lon.npy', help='Observation station Latitude and Longitude File')
parser.add_argument('--csta_test_path', type=str, default='obs_test_lat_lon.npy', help='Observation station Latitude and Longitude File')
parser.add_argument('--node_index_path', type=str, default='station_split.txt', help='Model Path')
parser.add_argument('--train_type',type=str,default='STKri',help='Kriging|Normal|ST|STKri')
parser.add_argument('--save_path', type=str, default='../../save/ChinaSouth', help='Result Save File Path')
parser.add_argument('--adj_path', type=str, default='sensor_graph/adj_mat.pkl', help='Observation Station Adjacency Matrix File Path')

parser.add_argument('--device',type=str,default='cuda:2',help='')
parser.add_argument('--runs', type=int, default=1, help='Number of Runs')
parser.add_argument('--M', type=int, default=24, help='Number of Runs')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--k', type=int, default=6,help='k Nearest Neighbors for Proxy Station')
parser.add_argument('--area',type=str,default='ChinaSouth',help='region')
parser.add_argument('--target', type=int, default=3, help='[u,v,msl,tmp]')
parser.add_argument('--d_align', type=int, default=16, help='')
parser.add_argument('--num_layer', type=int, default=2, help='Number of Encoder Layers')
parser.add_argument('--seq_len', type=int, default=24, help='Input Sequence Length')
parser.add_argument('--pre_len',type=int,default=24,help='Output Sequence Length')
parser.add_argument('--d_model',type=int,default=16,help='Embedding Dimension')
parser.add_argument('--in_dim',type=int,default=4,help='inputs dimension')
parser.add_argument('--out_dim',type=int,default=1,help='outputs dimension')
parser.add_argument('--pan_in_dim',type=int,default=4,help='inputs dimension')
parser.add_argument('--model',type=str,default='GenAISF',help='model type')
parser.add_argument('--ST_model',type=str,default='STDN',help='ST model type')
parser.add_argument('--lr', type=float, default=0.001, help='Training Learning Rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--num_nodes', type=int, default= 126,help='Number of Observation Stations')
parser.add_argument('--epochs', type=int, default=100, help='Training Epochs')
parser.add_argument('--print_every',type=int,default=10,help='')
parser.add_argument('--mark_dim',type=int,default=5,help='')
parser.add_argument('--gcn_bool',action='store_true',default=True,help='whether to add graph convolution layer')
parser.add_argument('--addaptadj',action='store_true',default=True,help='whether add adaptive adj')
parser.add_argument('--train_ratio',default=0.7,type=float,help='loss weight')
parser.add_argument('--gamma',default=0.05,type=float,help='loss weight')
args = parser.parse_args()

#路径更新
args.cpan_path=os.path.join(args.data_path,args.cpan_path)
args.csta_path=os.path.join(args.data_path,args.csta_path)
args.csta_train_path=os.path.join(args.data_path,args.csta_train_path)
args.csta_val_path=os.path.join(args.data_path,args.csta_val_path)
args.csta_test_path=os.path.join(args.data_path,args.csta_test_path)
args.node_index_path=os.path.join(args.data_path,args.node_index_path)
args.adj_path=os.path.join(args.data_path,args.adj_path)

device=torch.device(args.device)
# Create Model Save Path
if args.train_type=='STKri':
    run_dir=create_save_path(args.save_path, f'{args.model}_{args.ST_model}',args.train_type)
else:
    run_dir=create_save_path(args.save_path, args.model,args.train_type)

# Print Parameters and Save to Work Log
log_file_path = run_dir + '/work_log.txt'
log_file=print_args(args, log_file_path)

target_list=['u','v','msl','tmp']
target_name=target_list[args.target]
#Generate Predefined Graph
predefined_A = load_adj(pkl_filename = args.adj_path)
predefined_A = [torch.tensor(adj).to(device) for adj in predefined_A]
dataloader = load_dataset(args.data_path, args.target,args.batch_size, args)
obs_scaler,pan_scaler,target_scaler,target_pan_scaler = dataloader['obs_scaler'],dataloader['pan_scaler'],dataloader['target_scaler'],dataloader['target_pan_scaler']
train_num,val_num,test_num=dataloader['train_num'],dataloader['val_num'],dataloader['test_num']


# Initialize Training Model
engine = Trainer(args,predefined_A,target_scaler)

# Start Training
print("-------------------Start Training--------------------\n")
log_file.write("-------------------Start Training--------------------\n")

his_loss = []
minl = 1e5
epoch_best = -1
pbar = tqdm(range(args.epochs))
for epoch in pbar:
    train_metrics=[]
    #Shuffle Training Data
    dataloader['train_loader'].shuffle()
    for iter, (obs_his, obs_fut,pan_fut,csta,cpan,his_mark,fut_mark) in enumerate(dataloader['train_loader'].get_iterator()):
        obs_his = torch.Tensor(obs_his.astype(float)).to(device)# Tensor:(B,L,N,C)
        obs_fut = torch.Tensor(obs_fut.astype(float)).to(device) # Tensor:(B,L,N,C)
        pan_fut = torch.Tensor(pan_fut.astype(float)).to(device) # Tensor:(B,L,lat,lon,C)
        his_mark= torch.Tensor(his_mark.astype(float)).to(device)
        fut_mark = torch.Tensor(fut_mark.astype(float)).to(device)
        csta=csta.astype(float)
        cpan=cpan.astype(float)
        iter_per_epoch=math.ceil(dataloader['train_loader'].size / args.batch_size)
        metrics = engine.train(obs_his,obs_fut,pan_fut,csta,cpan,his_mark,fut_mark,args.target,epoch,iter,iter_per_epoch)
        train_metrics.append(metrics)

        if iter % args.print_every == 0:
            log = 'Iter: {:03d}, Train Loss: {:.4f}, train_mae  {:.4f}, train_rmse  {:.4f}, train_pear  {:.4f}, train_r  {:.4f}, train_smape  {:.4f}, train_fss  {:.4f}\n'
            print(log.format(iter, train_metrics[-1][0], train_metrics[-1][1], train_metrics[-1][2], train_metrics[-1][3], train_metrics[-1][4], train_metrics[-1][5], train_metrics[-1][6]), flush=True)
            log_file.write(log.format(iter, train_metrics[-1][0], train_metrics[-1][1], train_metrics[-1][2], train_metrics[-1][3], train_metrics[-1][4], train_metrics[-1][5], train_metrics[-1][6]))

    pbar.set_description("Training Processing %s" % int(epoch + 1))
    # Compute Average Metrics
    mtrain_metrics =np.mean(np.array(train_metrics),axis=0)

    # Start Validation
    val_metrics=[]
    for iter, (obs_his, obs_fut,pan_fut,csta,cpan,his_mark,fut_mark) in enumerate(dataloader['val_loader'].get_iterator()):
        obs_his = torch.Tensor(obs_his.astype(float)).to(device)  # Tensor:(B,C,N,L)
        obs_fut = torch.Tensor(obs_fut.astype(float)).to(device)  # Tensor:(B,C,N,L)
        pan_fut = torch.Tensor(pan_fut.astype(float)).to(device) # Tensor:(B,C,lat,lon,L)
        his_mark = torch.Tensor(his_mark.astype(float)).to(device)
        fut_mark = torch.Tensor(fut_mark.astype(float)).to(device)

        csta = csta.astype(float)
        cpan = cpan.astype(float)

        metrics = engine.eval(obs_his,obs_fut,pan_fut,csta,cpan,his_mark,fut_mark,args.target)
        val_metrics.append(metrics)

    # Compute Average Metrics
    mval_metrics =np.mean(np.array(val_metrics),axis=0)
    mval_loss=mval_metrics[0]
    log = 'Iter: {:03d}, valid Loss: {:.4f}, valid_mae  {:.4f}, valid_rmse  {:.4f}, valid_pear  {:.4f}, valid_r  {:.4f}, valid_smape  {:.4f}, valid_fss  {:.4f}\n'
    print(log.format(iter, mval_metrics[0], mval_metrics[1], mval_metrics[2], mval_metrics[3], mval_metrics[4],
                     mval_metrics[5], mval_metrics[6]), flush=True)
    his_loss.append(mval_loss)

    # Save the Model with the Lowest Validation Loss
    if mval_loss < minl:
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)  # 如果文件夹不存在则创建
        pth_path=run_dir+'/'+ target_name+'_'+ "best_epoch" + ".pth"
        torch.save(engine.model.state_dict(),pth_path)
        print('Model Saved')
        minl = mval_loss
        epoch_best = epoch

# Save the Epoch with the Lowest Loss
bestid = np.argmin(his_loss)
engine.model.load_state_dict(torch.load(pth_path, map_location='cpu'))
print("The valid loss on best model is {}, epoch:{}\n".format(str(round(his_loss[bestid], 4)), epoch_best))
log_file.write("The valid loss on best model is {}, epoch:{}\n".format(str(round(his_loss[bestid], 4)), epoch_best))
print("Training finished\n")

#Start testing
outputs = []
realy = torch.Tensor(dataloader['obs_fut_test'].astype(float)).to(device).permute(0, 3, 2, 1)#B,C,N,L

for iter, (obs_his, obs_fut,pan_fut,csta,cpan,his_mark,fut_mark) in enumerate(
        dataloader['test_loader'].get_iterator()):
    obs_his = torch.Tensor(obs_his.astype(float)).to(device)  # Tensor:(B,C,N,L)
    obs_fut = torch.Tensor(obs_fut.astype(float)).to(device)  # Tensor:(B,C,N,L)
    pan_fut = torch.Tensor(pan_fut.astype(float)).to(device)  # Tensor:(B,C,lat,lon,L)
    his_mark = torch.Tensor(his_mark.astype(float)).to(device)
    fut_mark = torch.Tensor(fut_mark.astype(float)).to(device)

    csta = csta.astype(float)
    cpan = cpan.astype(float)
    preds = engine.test(obs_his, obs_fut,  pan_fut, csta, cpan, his_mark, fut_mark, args.target,engine.model)

    outputs.append(preds)#B,1,N,L
yhat = torch.cat(outputs, dim=0)
yhat = yhat[:realy.size(0), ...]
if args.model=='Pangu':
    yhat_inv = target_pan_scaler.inverse_transform(yhat).permute(0, 3, 2, 1)  # B,L,N,1
else:
    yhat_inv = target_scaler.inverse_transform(yhat).permute(0, 3, 2, 1)#B,L,N,1
if args.train_type=='ST':
    train_N=int(args.num_nodes*0.7)
    yhat_data = yhat_inv.reshape(-1, train_N).cpu().numpy()
    save_experiment_result(args.model,args.ST_model, target_name, yhat_data, args.save_path, args.train_type)
    ground_truth = dataloader['obs_fut_test'][:, :, :train_N, args.target].astype(float).reshape(-1, train_N)
    save_experiment_result('GT', args.ST_model,target_name, ground_truth, args.save_path, args.train_type)
else:
    yhat_data=yhat_inv.reshape(-1,test_num).cpu().numpy()
    save_experiment_result(args.model, args.ST_model,target_name, yhat_data,args.save_path,args.train_type)
    ground_truth = dataloader['obs_fut_test'][:, :, :, args.target].astype(float).reshape(-1, test_num)
    save_experiment_result('GT', args.ST_model,target_name, ground_truth, args.save_path,args.train_type)

test_metrics= []
if args.train_type=='ST':
    test_num=int(args.num_nodes*0.7)
else:
    test_num=test_num

for node in range(test_num):
    pred = target_scaler.inverse_transform(yhat[:, :,node, :]).reshape(-1,1)#y_hat:B,1,N,L->pred:B,L
    real = realy[:, [args.target], node, :].reshape(-1,1)
    metrics = metric(pred, real)
    log = 'Evaluate best model on test data for station {:d}, test_mae:{:.4f}, test_rmse:{:.4f}, test_pear:{:.4f}, test_r:{:.4f}, test_smape:{:.4f}, test_fss:{:.4f}\n'
    print(log.format(node, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4],metrics[5]))
    log_file.write(log.format(node + 1, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4],metrics[5]))
    test_metrics.append(metrics)

mtest_metrics=np.mean(np.array(test_metrics),axis=0)
log = 'On average over {:} horizons, test_mae:{:.4f}, test_rmse:{:.4f}, test_pear:{:.4f}, test_r:{:.4f}, test_smape:{:.4f}, test_fss:{:.4f}\n'
print(log.format(test_num, mtest_metrics[0], mtest_metrics[1], mtest_metrics[2], mtest_metrics[3], mtest_metrics[4], mtest_metrics[5]))
log_file.write(log.format(test_num, mtest_metrics[0], mtest_metrics[1], mtest_metrics[2], mtest_metrics[3], mtest_metrics[4], mtest_metrics[5]))

