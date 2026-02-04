import numpy as np
import os
from scripts.utils import StandardScaler
import torch


def load_dataset(dataset_dir,target,batch_size,args):

    data={}
    for category in ['train','val','test']:
        cat_data=np.load(os.path.join(dataset_dir,category+'.npz'),allow_pickle=True)
        data['obs_his_'+category]=cat_data['obs_his'][:,:,:,:args.in_dim].astype(float)
        data['obs_fut_' + category] = cat_data['obs_fut'][:,:,:,:args.in_dim].astype(float)
        data['pan_fut_'+category]=np.transpose(cat_data['pan_fut'],(0,1,3,4,2))[:,:,:,:,:args.in_dim].astype(float)
        data['his_mark_'+category]=cat_data['his_mark'].astype(float)
        data['fut_mark_'+category]=cat_data['fut_mark'].astype(float)

    #标准化scaler计算
    obs_mean=[data['obs_his_train'][:,:,:,i].mean() for i in range(args.in_dim)]
    obs_std=[data['obs_his_train'][:,:,:,i].std() for i in range(args.in_dim)]
    obs_scaler=StandardScaler(mean=obs_mean,std=obs_std,device=args.device)
    pan_mean = [data['pan_fut_train'][:, :, :, :,i].mean() for i in range(args.in_dim)]
    pan_std = [data['pan_fut_train'][:, :, :,:, i].std() for i in range(args.in_dim)]
    pan_scaler = StandardScaler(mean=pan_mean, std=pan_std,device=args.device)
    target_mean=data['obs_his_train'][:,:,:,[target]].mean()
    target_std=data['obs_his_train'][:,:,:,[target]].std()
    target_scaler = StandardScaler(mean=target_mean, std=target_std,device=args.device)
    target_pan_mean=data['pan_fut_train'][:,:,:,:,[target]].mean()
    target_pan_std = data['pan_fut_train'][:,:,:,:,[target]].std()
    target_pan_scaler=StandardScaler(mean=target_pan_mean, std=target_pan_std,device=args.device)


    #对训练集/测试集/验证集的数据进行标准化操作
    for category in ['train','val','test']:
        data['obs_his_'+category]=obs_scaler.transform(data['obs_his_'+category])
        data['pan_fut_'+category]=pan_scaler.transform(data['pan_fut_'+category])

    #获取csta,cera,cpan
    csta_train=np.load(args.csta_train_path,allow_pickle=True)
    csta_val=np.load(args.csta_val_path,allow_pickle=True)
    csta_test=np.load(args.csta_test_path,allow_pickle=True)
    data['test_num']=csta_test.shape[0]
    data['train_num']=csta_train.shape[0]
    data['val_num']=csta_val.shape[0]
    csta=np.load(args.csta_path,allow_pickle=True)
    cpan=np.load(args.cpan_path,allow_pickle=True)


    data['train_loader']=DataLoader(data['obs_his_train'],
                                    data['obs_fut_train'],
                                    data['pan_fut_train'],
                                    csta_train,cpan,
                                    data['his_mark_train'],
                                    data['fut_mark_train'],
                                    batch_size)
    data['val_loader'] = DataLoader(data['obs_his_val'],
                                    data['obs_fut_val'],
                                    data['pan_fut_val'],
                                    csta_val, cpan,
                                    data['his_mark_val'],
                                    data['fut_mark_val'],
                                    batch_size)
    data['test_loader'] = DataLoader(data['obs_his_test'],
                                    data['obs_fut_test'],
                                    data['pan_fut_test'],
                                    csta_test,  cpan,
                                     data['his_mark_test'],
                                     data['fut_mark_test'],
                                     batch_size)

    data['obs_scaler']=obs_scaler
    data['pan_scaler']=pan_scaler
    data['target_scaler']=target_scaler
    data['target_pan_scaler']=target_pan_scaler
    return data

class DataLoader(object):
    def __init__(self,obs_his,obs_fut,pan_fut,csta,cpan,his_mark,fut_mark,batch_size,pad_with_last_sample=True):
        self.batch_size=batch_size
        self.current_ind=0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(obs_his) % batch_size)) % batch_size
            obs_his_padding = np.repeat(obs_his[-1:], num_padding, axis=0)
            obs_fut_padding = np.repeat(obs_fut[-1:], num_padding, axis=0)
            pan_fut_padding = np.repeat(pan_fut[-1:], num_padding, axis=0)
            his_mark_padding = np.repeat(his_mark[-1:], num_padding, axis=0)
            fut_mark_padding = np.repeat(fut_mark[-1:], num_padding, axis=0)

            obs_his = np.concatenate([obs_his, obs_his_padding], axis=0)
            obs_fut=np.concatenate([obs_fut,obs_fut_padding],axis=0)
            pan_fut = np.concatenate([pan_fut, pan_fut_padding], axis=0)
            his_mark = np.concatenate([his_mark, his_mark_padding], axis=0)
            fut_mark = np.concatenate([fut_mark, fut_mark_padding], axis=0)


        self.size=len(obs_his)
        self.num_batch=int(self.size//self.batch_size)
        self.obs_his=obs_his
        self.obs_fut=obs_fut
        self.pan_fut=pan_fut
        self.his_mark=his_mark
        self.fut_mark=fut_mark

        self.obs_his_s = obs_his
        self.obs_fut_s = obs_fut
        self.pan_fut_s = pan_fut
        self.his_mark_s = his_mark
        self.fut_mark_s = fut_mark

        self.csta=csta
        self.cpan=cpan

    def len(self):
        return self.num_batch


    def shuffle(self):
        self.original_indices=np.arange(self.size)
        permutation = torch.randperm(self.size)

        obs_his = self.obs_his[permutation]
        obs_fut = self.obs_fut[permutation]
        pan_fut = self.pan_fut[permutation]
        his_mark = self.his_mark[permutation]
        fut_mark = self.fut_mark[permutation]

        self.obs_his_s = obs_his
        self.obs_fut_s = obs_fut
        self.pan_fut_s = pan_fut
        self.his_mark_s = his_mark
        self.fut_mark_s = fut_mark
        self.shuffled_indices=permutation

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                obs_his_i = self.obs_his_s[start_ind: end_ind, ...]
                obs_fut_i = self.obs_fut_s[start_ind: end_ind, ...]
                pan_fut_i = self.pan_fut_s[start_ind: end_ind, ...]
                his_mark_i = self.his_mark[start_ind: end_ind, ...]
                fut_mark_i = self.fut_mark[start_ind: end_ind, ...]


                csta_i=self.csta
                cpan_i=self.cpan

                yield (obs_his_i,obs_fut_i,pan_fut_i,csta_i,cpan_i,his_mark_i,fut_mark_i)
                self.current_ind += 1
        return _wrapper()