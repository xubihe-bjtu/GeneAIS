import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from layers.TimeEncoder import TCN
from scripts.complex_tools import ComplexReLU,ComplexDropout,ComplexLayerNorm,ComplexSigmoid


class find_k_nearest_neighbors(nn.Module):
    def __init__(self, k,device):
        super(find_k_nearest_neighbors, self).__init__()
        self.k=k
        self.device=device

    def forward(self,obs_his,cobs,pan_fut,cpan):
        B,C,N,L = obs_his.shape

        pan_fut=pan_fut.reshape(B,C,-1,L)
        cpan_flat = cpan.reshape(-1, 2)  # (lat * lon, 2)

        nbrs_pan = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(cpan_flat)

        pan_k=[]
        cpan_k=[]
        for n in range(N):
            station_coord = np.array(cobs[n]).reshape(1,2)  # (2,)
            _, indices_pan = nbrs_pan.kneighbors(station_coord)
            indices_pan=torch.Tensor(indices_pan).to(self.device).long()
            pan_fut_n=pan_fut[:,:,indices_pan,:]#pan_fut:(B,C,1,k,L)
            cpan_n = torch.Tensor(cpan_flat[indices_pan.cpu(), :]).to(self.device)

            pan_k.append(pan_fut_n)
            cpan_k.append(cpan_n)
        pan_k=torch.cat(pan_k,dim=2)#pan_k:(B,C,N,k,L)
        if self.k!=1:
            cpan_k=torch.cat(cpan_k,dim=0)
        else:
            cpan_k=torch.cat(cpan_k,dim=0)
            cpan_k=cpan_k.reshape(N,2)
            cpan_k=cpan_k.unsqueeze(1)
        return pan_k,cpan_k

class frequency_feature_extractor(nn.Module):
    def __init__(self, args):
        super(frequency_feature_extractor, self).__init__()
        self.seq_len=args.seq_len
        self.d_model=16
        self.fc1=nn.Linear(self.seq_len//2+1, self.d_model).to(torch.cfloat)
        self.fc2=nn.Linear(self.d_model, self.seq_len//2+1).to(torch.cfloat)

    def forward(self,x):
        x=x.squeeze(-2)#B,C,N,L
        B, C,N,L = x.shape
        x=x.reshape(-1,N,L)
        x=torch.fft.rfft(x, dim=-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=torch.fft.irfft(x, dim=-1)
        x=x.reshape(B,C,N,L)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        hour_x = self.hour_embed(x[:, :, 4])
        weekday_x = self.weekday_embed(x[:, :, 3])
        day_x = self.day_embed(x[:, :, 2])
        month_x = self.month_embed(x[:, :, 1])
        return hour_x + weekday_x + day_x + month_x

class WFM_Correction(nn.Module):
    def __init__(self, args):
        super(WFM_Correction, self).__init__()
        self.d_model=16
        self.embed_type='fixed'
        self.freq='h'
        self.seq_len=args.seq_len
        self.in_dim=args.in_dim
        self.num_layers=2
        self.d_mark=16
        self.time_emb=TemporalEmbedding(self.d_mark, self.embed_type, self.freq)
        self.data_emb1=TCN(self.in_dim, self.d_model+self.d_mark, self.num_layers, kernel_size=3, max_dilation=16,d_model=self.d_model,d_mark=self.d_mark)
        self.data_emb2=TCN(self.in_dim, self.d_model+self.d_mark, self.num_layers, kernel_size=3, max_dilation=16,d_model=self.d_model,d_mark=self.d_mark)
        self.correct_fc1=nn.Conv2d(2*(self.d_model+self.d_mark),self.d_model+self.d_mark,kernel_size=(1, 1))
        self.correct_fc2=nn.Conv2d(self.d_model+self.d_mark,self.in_dim,kernel_size=(1, 1))

    def forward(self,x,tilde_pan,his_mark,fut_mark):
        B,C,N,L=x.shape
        his_mark_enc=self.time_emb(his_mark)#B,L,D
        fut_mark_enc=self.time_emb(fut_mark)#B,L,D
        x_emb,x_emb_h=self.data_emb1(x,his_mark_enc)#B,2D,N,L
        p_emb,p_emb_h=self.data_emb2(tilde_pan,fut_mark_enc)#B,2D,N,L
        xp_emb=torch.cat([x_emb,p_emb],dim=1)#B,2D,N,L
        hat_p=self.correct_fc2(self.correct_fc1(xp_emb))#B,C,N,L
        xp=torch.cat([x,hat_p],dim=-1)#B,C,N,2L
        xp_emb=torch.cat([x_emb,p_emb],dim=-1)#B,D,N,2L
        return xp_emb,xp,hat_p

class Complex_Attention(nn.Module):
    def __init__(self, args,M):
        super(Complex_Attention, self).__init__()
        self.M=M
        self.dropout = nn.Dropout(0.1)
        self.attn_mask = None
        self.d_model=args.d_model
        self.Q = nn.Linear(self.d_model, self.d_model)
        self.K = nn.Linear(self.d_model, self.d_model)
        self.V = nn.Linear(self.d_model, self.d_model)

    def forward(self,xp_emb,xp_temp_emb):
        # attn:Q(B,C,N,D),K,V(B,C,N,M,D)
        B,C,N,_=xp_emb.shape
        q = self.Q(xp_emb.reshape(-1, N, self.d_model))  # BC,N,d
        k = self.K(xp_temp_emb.reshape(-1, N * self.M, self.d_model))  # BC,NM,d
        v = self.V(xp_temp_emb.reshape(-1, N * self.M, self.d_model))  # BC,NM,d
        B, N, D = q.shape  # Queries shape
        _, NM, _ = k.shape  # Keys and Values shape

        scale = 1. / np.sqrt(D)
        scores = torch.einsum("bnd,bmd->bnm", q, k)  # shape: (B, N, NM)

        if self.attn_mask is not None:
            scores.masked_fill_(self.attn_mask == 0, -np.inf)  # -inf to mask out unwanted positions

        A = F.softmax(scale * scores, dim=-1)  # shape: (B, N, NM)
        #A = self.dropout(A)
        output = torch.einsum("bnm,bmd->bnd", A, v)  # shape: (B, N, D)
        return output,A

class Time_Attention(nn.Module):
    def __init__(self, args,M):
        super(Time_Attention, self).__init__()
        self.M=M
        self.dropout = nn.Dropout(0.1)
        self.seq_len=args.seq_len
        self.d_model=args.d_model
        self.Q = nn.Linear(2*self.seq_len, self.d_model)
        self.K = nn.Linear(2*self.seq_len, self.d_model)
        self.V = nn.Linear(2*self.seq_len, self.d_model)

    def forward(self,x,attn_mask):
        # attn:Q(B,C,N,D),K,V(B,C,N,M,D)
        B,D,N,_=x.shape
        q = self.Q(x.reshape(-1, N, 2*self.seq_len))  # BC,N,d
        k = self.K(x.reshape(-1, N, 2*self.seq_len))  # BC,N,d
        v = self.V(x.reshape(-1, N, 2*self.seq_len))  # BC,N,d
        B, N, D = q.shape  # Queries shape

        scale = 1. / np.sqrt(D)
        scores = torch.einsum("bnd,bmd->bnm", q, k)  # shape: (B, N, NM)
        scores = scores * scale
        temperature=0.05

        if attn_mask is not None:
            def _mask(scores, attn_mask):
                large_negative = -math.log(1e10)
                attention_mask = torch.where(attn_mask == 0, large_negative, 0)
                scores = scores * attn_mask.unsqueeze(0) + attention_mask.unsqueeze(0)
                return scores
            scores = _mask(scores, attn_mask)
        A = F.softmax(scores/temperature, dim=-1)  # shape: (B, N, NM)
        A = self.dropout(A)
        output = torch.einsum("bnm,bmd->bnd", A, v)  # shape: (B, N, D)
        return output,A



class Frequency_Aware_Delay_Attn(nn.Module):
    def __init__(self, args, M):
        super(Frequency_Aware_Delay_Attn, self).__init__()
        self.seq_len = 2*args.seq_len // 2 + 1  # rfft后长度
        self.device = args.device
        self.d_model = args.d_model
        self.M = M
        self.pred_len = args.pre_len
        # 复数操作
        self.relu=ComplexReLU()
        self.complex_dropout = ComplexDropout(0.1)
        self.layernorm=ComplexLayerNorm(self.d_model)
        self.sigmoid=ComplexSigmoid()
        # 1. 改进的频域模板初始化
        self.temp_real = nn.Parameter(torch.randn(M, self.seq_len))
        self.temp_imag = nn.Parameter(torch.randn(M, self.seq_len))

        # 2. 添加层归一化
        self.layernorm1 = nn.LayerNorm(self.d_model)
        self.layernorm2 = nn.LayerNorm(self.d_model)

        # 3. 改进的投影层，添加残差连接
        self.fc1 = nn.Linear(self.seq_len, self.d_model).to(torch.cfloat)
        self.fc2 = nn.Linear(self.d_model, self.d_model).to(torch.cfloat)
        self.fc3 = nn.Linear(self.d_model, self.seq_len).to(torch.cfloat)

        self.fc4 = nn.Linear(self.seq_len, self.d_model).to(torch.cfloat)
        self.fc5 = nn.Linear(self.d_model, self.d_model).to(torch.cfloat)
        # 4. 添加门控机制
        self.gate = nn.Linear(self.d_model * 2, self.d_model).to(torch.cfloat)

        # 5. 改进的注意力机制
        self.real_attn = Improved_Complex_Attention(args, M)
        self.imag_attn = Improved_Complex_Attention(args, M)

        # 6. 输出层改进
        self.output_proj = nn.Sequential(
            nn.Linear(2 * self.pred_len, 4 * self.pred_len),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(4 * self.pred_len, self.pred_len)
        )

        # 7. 频域权重学习
        self.freq_weights = nn.Parameter(torch.ones(self.seq_len))
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        for module in [self.fc1, self.fc2, self.fc3,self.fc4, self.fc5]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        # 频域模板初始化
        nn.init.normal_(self.temp_real, mean=0.0, std=0.02)
        nn.init.normal_(self.temp_imag, mean=0.0, std=0.02)

        # 频域权重初始化
        nn.init.constant_(self.freq_weights, 1.0)

    def forward(self, xp):

        B, C, N, L = xp.shape#B,C,N,2L

        xp_freq = torch.fft.rfft(xp, dim=-1)  # B,C,N,L+1

        freq_weights = torch.sigmoid(self.freq_weights)
        xp_freq = xp_freq * freq_weights.view(1, 1, 1, -1)#B,C<N<L+1

        temp_complex = torch.complex(self.temp_real, self.temp_imag)#M,L+1
        temp = temp_complex.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # 1,1,1,M,L+1
        temp = temp.repeat(B, C, N, 1, 1)  # B,C,N,M,L+1

        xp_temp = xp_freq.unsqueeze(-2) * temp  # B,C,N,M,L+1

        xp_temp_flat = xp_temp.reshape(-1, self.M, self.seq_len)
        xp_temp_emb = self.fc1(xp_temp_flat)  # (B*C*N), M, d_model
        xp_temp_emb = self.relu(xp_temp_emb)
        xp_temp_emb = self.fc2(xp_temp_emb)  # (B*C*N), M, d_model
        xp_temp_emb = xp_temp_emb.reshape(B, C, N, self.M, -1)

        xp_flat = xp_freq.reshape(-1, self.seq_len)
        xp_emb = self.fc4(xp_flat)  # (B*C*N), d_model
        xp_emb = self.relu(xp_emb)
        xp_emb = self.fc5(xp_emb)  # (B*C*N), d_model
        xp_emb = xp_emb.reshape(B, C, N, -1)#B,C,N,D

        xp_emb_real, xp_emb_imag = xp_emb.real, xp_emb.imag
        xp_temp_emb_real, xp_temp_emb_imag = xp_temp_emb.real, xp_temp_emb.imag

        output_real, A_real = self.real_attn(xp_emb_real, xp_temp_emb_real)
        output_imag, A_imag = self.imag_attn(xp_emb_imag, xp_temp_emb_imag)

        output_real = output_real + xp_emb_real
        output_imag = output_imag + xp_emb_imag

        output = torch.complex(output_real, output_imag)


        output = self.fc3(output)  # B,C,N,seq_len
        output = output.reshape(B, C, N, self.seq_len)

        output_time = torch.fft.irfft(output, n=L, dim=-1)#B,C,N,2L

        output_flat = output_time.reshape(-1, 2 * self.pred_len)
        output_final = self.output_proj(output_flat)
        output_final = output_final.reshape(B, C, N, self.pred_len)

        return output_final


class Improved_Complex_Attention(nn.Module):
    def __init__(self, args, M):
        super(Improved_Complex_Attention, self).__init__()
        self.d_model = args.d_model
        self.M = M
        self.scale = self.d_model ** -0.5

        # 多头注意力
        self.num_heads = 8
        self.head_dim = self.d_model // self.num_heads

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key_value):
        # query: B,C,N,d_model
        # key_value: B,C,N,M,d_model

        B, C, N, d_model = query.shape
        M = key_value.shape[3]

        Q = self.W_q(query).reshape(B, C, N, self.num_heads, self.head_dim)
        K = self.W_k(key_value).reshape(B, C, N, M, self.num_heads, self.head_dim)
        V = self.W_v(key_value).reshape(B, C, N, M, self.num_heads, self.head_dim)

        Q = Q.permute(0, 1, 3, 2, 4)  # B,C,num_heads,N,head_dim
        K = K.permute(0, 1, 4, 2, 3, 5)  # B,C,num_heads,N,M,head_dim
        V = V.permute(0, 1, 4, 2, 3, 5)  # B,C,num_heads,N,M,head_dim

        attn_scores = torch.matmul(Q, K.reshape(B,C,self.num_heads,-1,self.head_dim).transpose(-1, -2))#B,C,num_heads,N,N*M
        attn_scores = attn_scores * self.scale

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V.reshape(B,C,self.num_heads,-1,self.head_dim))#B,C,num_heads,N,head_dim
        output = output.permute(0, 1, 3, 2, 4).reshape(B, C, N, d_model)

        output = self.W_o(output)

        return output, attn_weights.mean(dim=2)


class Geo_Aware_Attn(nn.Module):
    def __init__(self, args):
        super(Geo_Aware_Attn, self).__init__()
        self.n_model = 32  # 增加维度
        self.sta_emb_1 = nn.Linear(2, self.n_model)
        self.node_emb1=nn.Linear(3*self.n_model, self.n_model)
        self.node_emb2 = nn.Linear(3 * self.n_model, self.n_model)
        self.seq_len = args.seq_len
        self.device = args.device
        self.attn = Time_Attention(args, self.seq_len)
        self.l_model = args.d_model
        self.d_mark = 16
        self.pred_len = args.pre_len
        self.in_dim = args.in_dim
        self.distance_scale = nn.Parameter(torch.tensor(2.0))  # 距离缩放参数

        # 数据相关性编码层
        self.d_model = args.d_model

        # 输出层
        self.fc1 = nn.Linear(self.l_model, self.pred_len)
        self.fc2 = nn.Linear(2 * self.d_mark, self.in_dim)

    def _compute_geo_graph(self, csta_tensor):

        N = csta_tensor.shape[0]
        nodevec = self.sta_emb_1(csta_tensor)#N,D
        node_sin=torch.sin(nodevec)
        node_cos=torch.cos(nodevec)
        node_raw=nodevec
        node_c=torch.cat([node_sin, node_cos,node_raw], dim=1)#N,3D
        emb1=F.relu(self.node_emb1(node_c))#N,D
        emb2=F.relu(self.node_emb2(node_c))

        # 归一化
        nodevec1 = F.normalize(emb1, p=2, dim=1)
        nodevec2 = F.normalize(emb2, p=2, dim=1)

        # 计算特征相似度
        geo_sim = torch.mm(nodevec1, nodevec2.T)

        # 计算地理距离（欧氏距离）
        lat = csta_tensor[:, 0]
        lon = csta_tensor[:, 1]

        # 向量化计算距离矩阵
        lat_diff = lat.unsqueeze(1) - lat.unsqueeze(0)
        lon_diff = lon.unsqueeze(1) - lon.unsqueeze(0)
        dist = torch.sqrt(lat_diff ** 2 + lon_diff ** 2 + 1e-8)

        # 更温和的距离相似度（距离越远相似度越低）
        distance_scale = torch.sigmoid(self.distance_scale) * 3
        dist_sim = torch.exp(-dist * distance_scale)

        # 结合方式：加权平均（避免乘法导致过度稀疏）
        combined_sim = geo_sim * 0.6 + dist_sim * 0.4

        # 添加全局连接基准（避免过度稀疏）
        global_bias = 0.05  # 小量全局连接
        combined_sim = combined_sim + global_bias

        # 确保对称性
        combined_sim = (combined_sim + combined_sim.T) / 2

        return combined_sim


    def forward(self, x_t, x, csta, cpan):
        B, D, N, L = x.shape

        # 2. 地理图计算
        def robust_min_max_normalization(csta, cpan):
            lat_values = cpan[:, :, 0]
            lon_values = cpan[:, :, 1]
            max_lat, min_lat = lat_values.max(), lat_values.min()
            max_lon, min_lon = lon_values.max(), lon_values.min()
            range_lat = max_lat - min_lat + 1e-8
            range_lon = max_lon - min_lon + 1e-8
            csta_min = np.array([min_lat, min_lon])
            normalized = (csta - csta_min) / np.array([range_lat, range_lon])
            return normalized

        csta_norm = robust_min_max_normalization(csta, cpan)
        csta_tensor = torch.tensor(csta_norm, dtype=torch.float32).to(self.device)
        geo_graph = self._compute_geo_graph(csta_tensor)

        t, t_attn = self.attn(x, geo_graph)

        output = self.fc1(t).reshape(B, D, N, self.pred_len).permute(0, 3, 2, 1).reshape(-1, N, D)
        output = self.fc2(output).reshape(B, self.pred_len, N, self.in_dim).permute(0, 3, 2, 1)

        return output, t_attn, geo_graph


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.in_dim=args.in_dim
        self.target=args.target
        self.device=args.device
        self.M=args.M
        self.pred_len=args.pre_len
        self.find_nearest_neighbors = find_k_nearest_neighbors(1, self.device)
        self.frequency_feature_extractor=frequency_feature_extractor(args)
        self.wfm_correction=WFM_Correction(args)
        self.frequency_awre_delay_attn=Frequency_Aware_Delay_Attn(args,self.M)
        self.geo_aware_attn=Geo_Aware_Attn(args)
        self.predict_layer=nn.Conv2d(2*self.in_dim,1,(1,1))
        self.predict_layer_1=nn.Conv2d(self.in_dim,1,(1,1))


    def forward(self,obs_his, pan_fut, csta, cpan,his_mark, fut_mark, adj, unknown_nodes, epoch, train):
        '''
        Input:obs_his train:(B,L,N_train,C) /obs_his val:(B,L,N_val,C) /obs_his test:(B,L,N_test,C)
              pan_fut:(B,L,lat,lon,C) [pan_fut_1 train:(B,L,N_train,C)]
              csta:(N,2)
              cpan:(lat,lon,2)
              his_mark/fut_mark:B,L,5
        '''
        obs_his = obs_his.permute(0,3,2,1)#B,C,N,L
        pan_fut = pan_fut.permute(0,4,2,3,1)#B,C,lat,lon,L
        pan_fut_1,c_pan_1=self.find_nearest_neighbors(obs_his,csta,pan_fut,cpan)
        tilde_pan=self.frequency_feature_extractor(pan_fut_1)#B,C,N,L
        xp_emb,xp,hat_p=self.wfm_correction(obs_his,tilde_pan,his_mark,fut_mark)#xp_emb:(32,32,256,48),xp:(32,4,256,48),hat_p:(32,4,256,24)
        hat_p = self.predict_layer_1(hat_p)  # B,1,N,L
        #(xp:B,C,N,2L)
        f=self.frequency_awre_delay_attn(xp)#B,C,N,L
        xp_target=xp[:,[self.target],:,:]
        t,t_attn,final_A=self.geo_aware_attn(xp_target,xp_emb,csta,cpan)#B,C,N,L
        z=torch.cat([f,t],dim=1)
        output=self.predict_layer(z)
        output=output.squeeze(1).transpose(1, 2)#B,L,N
        hat_p=hat_p.squeeze(1).transpose(1, 2)
        return output,hat_p