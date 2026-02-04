import torch.optim as optim
from scripts.metric import *
from models import GeneAIS
from scripts.utils import *

class Trainer():
    def __init__(self,args,predefined_A,target_scaler):
        self.model_dict = {'GeneAIS':GeneAIS}
        self.predefined_A = predefined_A

        target_list = ['u', 'v', 'msl', 'tmp']
        target_name = target_list[args.target]

        if args.train_type=='STKri':
            self.ST_model=self._build_model(args,args.ST_model).to(args.device)
            ST_model_path=f'../../save/{args.area}/ST/{args.ST_model}/{target_name}_best_epoch.pth'
            checkpoint = torch.load(ST_model_path)
            self.ST_model.load_state_dict(checkpoint)
            for param in self.ST_model.parameters():
                param.requires_grad = False

        self.model = self._build_model(args, args.model).to(args.device)
        if len(list(self.model.parameters())) != 0:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.loss = mae
        self.res_scaler = target_scaler
        self.model_type=args.model
        self.device=args.device
        self.gamma=args.gamma
        self.train_ratio = args.train_ratio
        self.adj_mx = load_raw_adj(pkl_filename=args.adj_path)
        self.train_index, self.val_index, self.test_index = load_station_splits(args.node_index_path)
        self.valid_all_index = torch.cat((torch.tensor(self.train_index), torch.tensor(self.val_index)), 0)
        self.test_all_index = torch.cat((torch.tensor(self.train_index), torch.tensor(self.test_index)), 0)
        self.matrix_transform = 'original'
        self.train_mx, _ = adj_transform(adj_node_index(self.adj_mx, self.train_index), self.matrix_transform)
        self.train_N=self.train_mx[0].shape[0]
        self.val_mx, _ = adj_transform(adj_node_index(self.adj_mx, self.valid_all_index), self.matrix_transform)
        self.test_mx, _ = adj_transform(adj_node_index(self.adj_mx, self.test_all_index), self.matrix_transform)
        self.batch_size = args.batch_size
        self.train_mx=torch.tensor(self.train_mx).repeat(self.batch_size, 1, 1)
        self.val_mx = torch.tensor(self.val_mx).repeat(self.batch_size, 1, 1)
        self.test_mx = torch.tensor(self.test_mx).repeat(self.batch_size, 1, 1)
        self.batch_size = args.batch_size
        self.target = args.target
        self.area=args.area
        self.train_type=args.train_type
        self.pre_len=args.pre_len


    def _build_model(self,args,model):
        model=self.model_dict[model].Model(args).float()
        return model

    def train(self, obs_his,obs_fut,pan_fut,csta,cpan,his_mark,fut_mark,target,epoch,iter_index,iter_per_epoch):
        '''
        :param obs_his:B,L_in,N_train,C
        :param obs_fut:B,L_out,N_train,C
        :param pan_fut:B,L_out,lat,lon,C
        :param csta:N_train,2
        :param cpan:lat,lon,2
        :param his_mark:B,L_in,5
        :param fut_mark:B,L_out,5
        :param target:1([0,1,2,3]->[u,v,msl,tmp])
        :param epoch:int(epoch index)
        :param iter_index:int(iter_index index)
        :param iter_per_epoch:46
        :return:
        '''

        self.model.train()
        if len(list(self.model.parameters())) != 0:
            self.optimizer.zero_grad()
        if self.train_type=='Kriging':
            iter_num = epoch  * iter_per_epoch + iter_index
            masked_node_num=int(len(self.train_index)*(1-self.train_ratio))
            train_mx=self.train_mx[0]#N_train,N_train
            index = np.random.permutation(len(self.train_index))
            train_mx = adj_node_index(train_mx, index)
            obs_his2=obs_his[:, :, index, :]
            obs_fut2=obs_fut[:, :, index, :]
            csta2=csta[index, :]
            data_ones=torch.ones_like(obs_his)
            real_value=torch.ones(self.batch_size,self.pre_len,masked_node_num)
            unknown_nodes=torch.zeros(self.batch_size,train_mx.shape[0])
            random_recoder,matrix_ls=[],[]
            for b in range(self.batch_size):
                random_select_node=np.random.choice(index,masked_node_num,replace=False)
                data_ones[b,:,random_select_node,:]=0
                real_value[b]=obs_fut2[b,:,random_select_node,self.target]
                unknown_nodes[b,random_select_node]=1
                random_recoder.append(random_select_node)
                matrix_ls.append(torch.FloatTensor(train_mx))
            matrix=torch.stack(matrix_ls,dim=0)#B,N_train,N_train
            obs_his_input=obs_his2*data_ones
            batch_random_nodes=np.stack(random_recoder,axis=0)
            if self.model_type=='GeneAIS':
                output,hat_p = self.model(obs_his=obs_his_input,
                                    pan_fut=pan_fut,
                                    csta=csta2,
                                    cpan=cpan,
                                    his_mark=his_mark,
                                    fut_mark=fut_mark,
                                    adj=matrix,
                                    unknown_nodes=unknown_nodes,
                                    epoch=epoch,
                                    train=True)
                output = output[torch.arange(self.batch_size)[:, None], :, batch_random_nodes].transpose(1, 2)
                hat_p = hat_p[torch.arange(self.batch_size)[:, None], :, batch_random_nodes].transpose(1, 2)
                predict = self.res_scaler.inverse_transform(output)
                hat_p_reverse = self.res_scaler.inverse_transform(hat_p)
                real = real_value.to(predict.device)
                loss = self.loss(predict, real, 0.0)+self.gamma*self.loss(hat_p_reverse, real, 0.0)
            else:
                output=self.model(obs_his=obs_his_input,
                                  pan_fut=pan_fut,
                                  csta=csta2,
                                  cpan=cpan,
                                  his_mark=his_mark,
                                  fut_mark=fut_mark,
                                  adj=matrix,
                                  unknown_nodes=unknown_nodes,
                                  epoch=epoch,
                                  train=True)
                output=output[torch.arange(self.batch_size)[:, None], :, batch_random_nodes].transpose(1, 2)
                predict = self.res_scaler.inverse_transform(output)
                real=real_value.to(predict.device)
                loss = self.loss(predict, real, 0.0)
            if self.model_type=='STAGANN':
                l1_regularization = torch.tensor(0.).to(loss.device)
                for param in self.model.parameters():
                    l1_regularization += torch.norm(param, 1)
                loss=loss+0.0001*l1_regularization
        elif self.train_type=='Normal' or self.train_type=='ST':
            if self.model_type=='GeneAIS':
                output,hat_p = self.model(obs_his=obs_his,
                                    pan_fut=pan_fut,
                                    csta=csta,
                                    cpan=cpan,
                                    his_mark=his_mark,
                                    fut_mark=fut_mark,
                                    adj=self.train_mx,
                                    unknown_nodes=None,
                                    epoch=epoch,
                                    train=True)  # B,L,N
                real = obs_fut[:, :, :, [self.target]]  # B,L,N,1
                predict = self.res_scaler.inverse_transform(output).unsqueeze(-1)
                hat_p_reverse=self.res_scaler.inverse_transform(hat_p).unsqueeze(-1)
                loss = self.loss(predict, real, 0.0)+self.gamma*self.loss(hat_p_reverse, real, 0.0)
            else:
                output = self.model(obs_his=obs_his,
                                    pan_fut=pan_fut,
                                    csta=csta,
                                    cpan=cpan,
                                    his_mark=his_mark,
                                    fut_mark=fut_mark,
                                    adj=self.train_mx,
                                    unknown_nodes=None,
                                    epoch=epoch,
                                    train=True)#B,L,N
                real = obs_fut[:,:,:,[self.target]]#B,L,N,1
                predict = self.res_scaler.inverse_transform(output).unsqueeze(-1)
                loss = self.loss(predict, real, 0.0)
                if self.model_type=='STAGANN':
                    l1_regularization = torch.tensor(0.).to(loss.device)
                    for param in self.model.parameters():
                        l1_regularization += torch.norm(param, 1)
                    loss = loss + 0.0001*l1_regularization
        elif self.train_type=='STKri':
            with torch.no_grad():
                pre_fut = self.ST_model(obs_his=obs_his,
                                pan_fut=pan_fut,
                                csta=csta,
                                cpan=cpan,
                                his_mark=his_mark,
                                fut_mark=fut_mark,
                                adj=self.train_mx,
                                unknown_nodes=None,
                                epoch=epoch,
                                train=False)
                pre_fut=pre_fut.unsqueeze(-1).repeat(1, 1, 1, 4)#B,L,N,4
            iter_num = epoch  * iter_per_epoch + iter_index
            masked_node_num=int(len(self.train_index)*(1-self.train_ratio))
            train_mx=self.train_mx[0]#N_train,N_train
            index = np.random.permutation(len(self.train_index))
            train_mx = adj_node_index(train_mx, index)
            data2=pre_fut[:, :, index, :]
            obs_fut2=obs_fut[:, :, index, :]
            csta2=csta[index, :]
            data_ones=torch.ones_like(obs_his)
            real_value=torch.ones(self.batch_size,self.pre_len,masked_node_num)
            unknown_nodes=torch.zeros(self.batch_size,train_mx.shape[0])
            random_recoder,matrix_ls=[],[]
            for b in range(self.batch_size):
                random_select_node=np.random.choice(index,masked_node_num,replace=False)
                data_ones[b,:,random_select_node,:]=0
                real_value[b]=obs_fut2[b,:,random_select_node,self.target]
                unknown_nodes[b,random_select_node]=1
                random_recoder.append(random_select_node)
                matrix_ls.append(torch.FloatTensor(train_mx))
            matrix=torch.stack(matrix_ls,dim=0)#B,N_train,N_train
            obs_his_input=data2*data_ones
            batch_random_nodes=np.stack(random_recoder,axis=0)
            output=self.model(obs_his=obs_his_input,
                              pan_fut=pan_fut,
                              csta=csta2,
                              cpan=cpan,
                              his_mark=his_mark,
                              fut_mark=fut_mark,
                              adj=matrix,
                              unknown_nodes=unknown_nodes,
                              epoch=epoch,
                              train=True)
            output=output[torch.arange(self.batch_size)[:, None], :, batch_random_nodes].transpose(1, 2)
            predict = self.res_scaler.inverse_transform(output)
            real=real_value.to(predict.device)
            loss = self.loss(predict, real, 0.0)
            if self.model_type=='STAGANN':
                l1_regularization = torch.tensor(0.).to(loss.device)
                for param in self.model.parameters():
                    l1_regularization += torch.norm(param, 1)
                loss=loss+0.0001*l1_regularization
        if len(list(self.model.parameters())) != 0:
            loss.backward()
            self.optimizer.step()
        mae, rmse, pear, r, smape, fss=metric(predict, real)
        return loss.item(), mae, rmse,pear,r,smape,fss


    def eval(self,obs_his,obs_fut,pan_fut,csta,cpan,his_mark,fut_mark,target):
        self.model.eval()
        if self.train_type=='Kriging':
            valid_mx=self.val_mx
            real_index=np.arange(self.valid_all_index.shape[0])
            if self.model_type=='GeneAIS':
                output,hat_p = self.model(obs_his=obs_his,
                                    pan_fut=pan_fut,
                                    csta=csta,
                                    cpan=cpan,
                                    his_mark=his_mark,
                                    fut_mark=fut_mark,
                                    adj=valid_mx,
                                    unknown_nodes=None,
                                    epoch=1,
                                    train=False)
                predict = self.res_scaler.inverse_transform(output)
                hat_p_reverse=self.res_scaler.inverse_transform(hat_p)
                real = obs_fut[:, :, :, self.target].to(predict.device)
                loss = self.loss(predict, real, 0.0)+self.gamma*self.loss(hat_p_reverse, real, 0.0)
            else:
                output = self.model(obs_his=obs_his,
                                    pan_fut=pan_fut,
                                    csta=csta,
                                    cpan=cpan,
                                    his_mark=his_mark,
                                    fut_mark=fut_mark,
                                    adj=valid_mx,
                                    unknown_nodes=None,
                                    epoch=1,
                                    train=False)
                predict = self.res_scaler.inverse_transform(output)
                real = obs_fut[:,:,:,self.target].to(predict.device)
                loss = self.loss(predict, real, 0.0)
        elif self.train_type=='Normal':
            if self.model_type=='GeneAIS':
                output,hat_p = self.model(obs_his=obs_his,
                                    pan_fut=pan_fut,
                                    csta=csta,
                                    cpan=cpan,
                                    his_mark=his_mark,
                                    fut_mark=fut_mark,
                                    adj=self.val_mx,
                                    unknown_nodes=None,
                                    epoch=1,
                                    train=False)
                real = obs_fut[:, :, :, [target]]
                predict = self.res_scaler.inverse_transform(output).unsqueeze(-1)
                hat_p_reverse=self.res_scaler.inverse_transform(hat_p).unsqueeze(-1)
                loss = self.loss(predict, real, 0.0)+self.gamma*self.loss(hat_p_reverse, real, 0.0)
            else:
                output = self.model(obs_his=obs_his,
                                    pan_fut=pan_fut,
                                    csta=csta,
                                    cpan=cpan,
                                    his_mark=his_mark,
                                    fut_mark=fut_mark,
                                    adj=self.val_mx,
                                    unknown_nodes=None,
                                    epoch=1,
                                    train=False)
                real = obs_fut[:, :, :, [target]]
                predict = self.res_scaler.inverse_transform(output).unsqueeze(-1)
                loss = self.loss(predict, real, 0.0)
        elif self.train_type=='ST':
            obs_his=obs_his[:,:,:self.train_N,:]#B,L,N_train,C
            csta=csta[:self.train_N]#N_train,2
            obs_fut=obs_fut[:,:,:self.train_N,:]
            output = self.model(obs_his=obs_his,
                                pan_fut=pan_fut,
                                csta=csta,
                                cpan=cpan,
                                his_mark=his_mark,
                                fut_mark=fut_mark,
                                adj=self.train_mx,
                                unknown_nodes=None,
                                epoch=1,
                                train=False)
            real = obs_fut[:, :, :, [target]]
            predict = self.res_scaler.inverse_transform(output).unsqueeze(-1)
            loss = self.loss(predict, real, 0.0)
        elif self.train_type=='STKri':
            with torch.no_grad():
                obs_his = obs_his[:, :, :self.train_N, :]  # B,L,N_train,C
                csta_ST = csta[:self.train_N]  # N_train,2
                pre_fut = self.ST_model(obs_his=obs_his,
                                pan_fut=pan_fut,
                                csta=csta_ST,
                                cpan=cpan,
                                his_mark=his_mark,
                                fut_mark=fut_mark,
                                adj=self.train_mx,
                                unknown_nodes=None,
                                epoch=1,
                                train=False)
                pre_fut=pre_fut.unsqueeze(-1).repeat(1, 1, 1, 4)#B,L,N,4
            none_fut=torch.zeros(pre_fut.shape[0], pre_fut.shape[1], len(self.val_index), 4).to(self.device)
            val_data=torch.cat([pre_fut, none_fut], dim=-2)
            valid_mx=self.val_mx
            real_index=np.arange(self.valid_all_index.shape[0])
            output = self.model(obs_his=val_data,
                                pan_fut=pan_fut,
                                csta=csta,
                                cpan=cpan,
                                his_mark=fut_mark,
                                fut_mark=fut_mark,
                                adj=valid_mx,
                                unknown_nodes=None,
                                epoch=1,
                                train=False)
            predict = self.res_scaler.inverse_transform(output)
            real = obs_fut[:,:,:,self.target].to(predict.device)
            loss = self.loss(predict, real, 0.0)
        mae, rmse, pear, r, smape, fss = metric(predict, real)
        return loss.item(), mae, rmse, pear, r, smape, fss

    def test(self,obs_his,obs_fut,pan_fut,csta,cpan,his_mark,fut_mark,target,test_model):
        with torch.no_grad():
            if self.train_type=='Kriging':
                test_mx=self.test_mx
                real_index=np.arange(self.test_all_index.shape[0])
                if self.model_type=='GeneAIS':
                    output ,_= self.model(obs_his=obs_his,
                                        pan_fut=pan_fut,
                                        csta=csta,
                                        cpan=cpan,
                                        his_mark=his_mark,
                                        fut_mark=fut_mark,
                                        adj=test_mx,
                                        unknown_nodes=real_index,
                                        epoch=1,
                                        train=False)
                    preds = output.unsqueeze(1).transpose(-1, -2)  # B,1,N,L
                else:
                    output = self.model(obs_his=obs_his,
                                        pan_fut=pan_fut,
                                        csta=csta,
                                        cpan=cpan,
                                        his_mark=his_mark,
                                        fut_mark=fut_mark,
                                        adj=test_mx,
                                        unknown_nodes=real_index,
                                        epoch=1,
                                        train=False)
                    preds=output.unsqueeze(1).transpose(-1, -2)#B,1,N,L
            elif self.train_type=='Normal':
                if self.model_type=='GeneAIS':
                    preds,_ = test_model(obs_his=obs_his,
                                       pan_fut=pan_fut,
                                       csta=csta,
                                       cpan=cpan,
                                       his_mark=his_mark,
                                       fut_mark=fut_mark,
                                       adj=self.test_mx,
                                       unknown_nodes=None,
                                       epoch=1,
                                       train=False)
                    preds = preds.unsqueeze(1).transpose(-1, -2)
                else:
                    preds = test_model(obs_his=obs_his,
                                        pan_fut=pan_fut,
                                        csta=csta,
                                        cpan=cpan,
                                        his_mark=his_mark,
                                        fut_mark=fut_mark,
                                        adj=self.test_mx,
                                        unknown_nodes=None,
                                        epoch=1,
                                        train=False)
                    preds=preds.unsqueeze(1).transpose(-1, -2)
            elif self.train_type=='ST':
                obs_his = obs_his[:, :, :self.train_N, :]  # B,L,N_train,C
                csta = csta[:self.train_N]  # N_train,2
                preds = test_model(obs_his=obs_his,
                                   pan_fut=pan_fut,
                                   csta=csta,
                                   cpan=cpan,
                                   his_mark=his_mark,
                                   fut_mark=fut_mark,
                                   adj=self.train_mx,
                                   unknown_nodes=None,
                                   epoch=1,
                                   train=False)
                preds = preds.unsqueeze(1).transpose(-1, -2)
            elif self.train_type=='STKri':
                with torch.no_grad():
                    obs_his = obs_his[:, :, :self.train_N, :]  # B,L,N_train,C
                    csta_ST = csta[:self.train_N]  # N_train,2
                    pre_fut = self.ST_model(obs_his=obs_his,
                                            pan_fut=pan_fut,
                                            csta=csta_ST,
                                            cpan=cpan,
                                            his_mark=his_mark,
                                            fut_mark=fut_mark,
                                            adj=self.train_mx,
                                            unknown_nodes=None,
                                            epoch=1,
                                            train=False)
                    pre_fut = pre_fut.unsqueeze(-1).repeat(1, 1, 1, 4)  # B,L,N,4
                none_fut = torch.zeros(pre_fut.shape[0], pre_fut.shape[1], len(self.test_index), 4).to(self.device)
                test_data = torch.cat([pre_fut, none_fut], dim=-2)
                test_mx=self.test_mx
                real_index=np.arange(self.test_all_index.shape[0])
                output = self.model(obs_his=test_data,
                                    pan_fut=pan_fut,
                                    csta=csta,
                                    cpan=cpan,
                                    his_mark=fut_mark,
                                    fut_mark=fut_mark,
                                    adj=test_mx,
                                    unknown_nodes=real_index,
                                    epoch=1,
                                    train=False)
                preds=output.unsqueeze(1).transpose(-1, -2)#B,1,N,L
        return preds
