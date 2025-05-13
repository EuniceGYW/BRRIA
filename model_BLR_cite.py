import argparse
import numpy as np
import copy

from utils import load_index, load_citeseer
from attack_utils_blr import get_vic_idx, get_inj_idx, compute_AV_dict, attack_citeseer_data, compute_feature_importance, sample_features
from model_gat import GAT_Model
from attack_blr import att_inj_features, edge_attacks
from models import GCN
import time
import torch
import torch.nn as nn 
import torch.nn.functional as F
from sklearn import linear_model
import pickle 

 

class BayesianLinearRegression():
    def __init__(self, idx, args, nclass=6):
        super(BayesianLinearRegression, self).__init__()
        
        self.nclass = nclass
        checkpoint = torch.load(r'./code/JESTIE/BLA/BLA/BLR_cite.pth')
        self.fc_layers_weight = torch.t(checkpoint['fc_layers.0.weight']).cpu()
        self.fc_layers_bias = checkpoint['fc_layers.0.bias'].cpu()
       
        self.brg = ([linear_model.BayesianRidge(max_iter=300) for _ in range(nclass)])
       
        # 全连接
        self.idx = idx
        self.args = args


    def fit(self, adj_matrix, x, label_y, idx):
        self.idx = torch.cat((self.idx, idx), -1)	
        
        idx_fit = torch.unique(self.idx) 
        brg_in = torch.mm(x[idx_fit,:], self.fc_layers_weight) 
        brg_in = self.adj_weighted_mean(brg_in, (adj_matrix[idx_fit,:])[:,idx_fit])

        self.brg_fit = []
        for i, brg in enumerate(self.brg):
    
            y_0 = brg.fit(brg_in, (label_y[idx_fit,:])[:,i])
            self.brg_fit.append(y_0)
            with open(r"./code/JESTIE/BLA/BLA/models/model_cite_{}.pkl".format(i), "wb") as f:
 
                pickle.dump(brg, f)      #以二进制可写模式打开

                
        
   







    
    def forward(self, adj_matrix, x, idx):
        self.idx = torch.cat((self.idx, idx), -1)	
        idx_forward = torch.unique(self.idx)   
        brg_in = torch.mm(x[idx_forward,:], self.fc_layers_weight) #(self.fc_layers_bias)

        brg_in = self.adj_weighted_mean(brg_in, (adj_matrix[idx_forward,:])[:,idx_forward])
        
        for i in range(self.nclass):
            with open(r"./code/JESTIE/BLA/BLA/models/model_cite_{}.pkl".format(i), "rb") as f:
                model = pickle.load(f)
            y_mean = model.predict(brg_in,return_std=False)
            

            if i == 0:
                y = y_mean
               

            else:    
                y = np.vstack((y,y_mean))

        y = torch.from_numpy(np.transpose(y))
        output_all = torch.zeros([adj_matrix.size(0),y.size(1)],dtype=torch.double)
        output_all[idx_forward] = y
        output_y = output_all[idx]

        return output_y
    

    def query(self, adj_matrix, x, idx, labels, labels_inject):
        self.idx = torch.cat((self.idx, idx), 0)	
        idx_forward = torch.unique(self.idx)   
        brg_in = torch.mm(x[idx_forward,:], self.fc_layers_weight) #(self.fc_layers_bias)

        brg_in = self.adj_weighted_mean(brg_in, (adj_matrix[idx_forward,:])[:,idx_forward])
        
        for i in range(self.nclass):
            with open(r"./code/JESTIE/BLA/BLA/models/model_cite_{}.pkl".format(i), "rb") as f:
                model = pickle.load(f)
            y_mean = model.predict(brg_in,return_std=False)
            

            if i == 0:
                y = y_mean
               

            else:    
                y = np.vstack((y,y_mean))

        y = torch.from_numpy(np.transpose(y))
        output_all = torch.zeros([adj_matrix.size(0),y.size(1)],dtype=torch.double)
        output_all[idx_forward] = y
        output_y = output_all[idx]
        best_second_class_value, best_second_class_indices = torch.max((output_y - 1000*labels_inject[idx]),dim=1)
        classification_margins = best_second_class_value - output_y[:,labels[idx]]
        return classification_margins
    
    def adj_weighted_mean(self, x, adj_matrix):
        x_weighted = torch.matmul(adj_matrix, x)
        return x_weighted / adj_matrix.sum(dim=-1, keepdim=True)
    


def accuracy(output, labels):
    preds = output.max(-1)[1].type_as(labels)  # 输出中的最大值的索引值（第几行），且与labels类型相同
    correct = preds.eq(labels).double()  # 判断与labels的值是否相同，相同为double型的1，否则为0。
    correct = correct.sum()  # 把所有比较结果加到一起
    return correct / len(labels)  # 除以与labels的长度，即参与判断的节点个数，得到accuracy

def set_seed(seed):  # 随机数种子，不用理解，都这么用
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0") 
    parser.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
    parser.add_argument("--random_seed", type=int, default=3440)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--attention_heads", type=int, default=7)  # 多头数，这个超参数是需要通过尝试调整的。
    parser.add_argument("--patient", type=int, default=100)

    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--hidden_layers", type=int, default=6)
    parser.add_argument("--hidden_layers_gcn", type=int, default=16)
    parser.add_argument('--dropout_gcn', type=float, default=0.7,help='Dropout rate (1 - keep probability).')
    
    parser.add_argument("--txt_log_path", type=str, default=r"./code/JESTIE/BLA/BLA/BLR_cite_gcn.txt")  # 打印文件训练集和验证集的结果
    parser.add_argument("--txt_attack_path", type=str, default=r"./code/JESTIE/BLA/BLA/BLA_cite_gcn.txt")  # 打印文件训练集和验证集的结果
    #parser.add_argument("--model_save_path", type=str, default=r"./pygcn/pygcn_print_model.ckpt")  # 保存模型的权重
    parser.add_argument("--model_save_path", type=str, default=r"./code/JESTIE/BLA/BLA/BLR_cite.pth")  # 保存模型的权重
    parser.add_argument("--GPR_model_save_path", type=str, default=r"./code/JESTIE/BLA/BLA/BLR_cite_model.pth")  # 保存模型的权重


    parser.add_argument("--nhid", type=int, default=256)  # 隐藏层输出维度，这个超参数是需要通过尝试调整的。
 
    parser.add_argument("--dropout", type=int, default=0.8)  # 经验/可以试一试?
    parser.add_argument("--dropout_gat", type=int, default=0.6)  # 经验/可以试一试?
    parser.add_argument("--alpha", type=int, default=0.2)  # 经验/可以试一试?LeakyReLU的负值部分的斜率。
    
    parser.add_argument("--theta_inj", type=float, default=0.1,  help="query ratio")  # 0<theta<1，越小说明target class的特征在query时更加重要。
    parser.add_argument("--theta_vic", type=float, default=0.1,  help="query ratio")  # 0<theta<1，越小说明victim class的特征在query时更加重要。
    parser.add_argument("--alpha_cluster", type=float, default=1.0,  help="query ratio")
    parser.add_argument("--lambdaN", type=float, default=0.0,  help="trade-off parameter between loss function of victim nodes and protected nodes.")

   
    # attack
    parser.add_argument("--injection_num", type=float, default=5)  # 注入节点数
    parser.add_argument("--victim_num", type=float, default=10)  # 受害者节点数
    parser.add_argument("--mini_batch", type=float, default=20)  # 每次的攻击数
    parser.add_argument("--query_num", type=float, default=100)  # 询问总次数
    parser.add_argument("--budget", type=float, default=100)  # 更改特征及连边总数
    parser.add_argument("--init_edges", type=float, default=2)  # 更改特征及连边总数
    parser.add_argument("--victim_class", type=float, default=5)  # 更改特征及连边总数
    parser.add_argument("--target_class", type=float, default=0)  # 更改特征及连边总数

    return parser.parse_args()
args = parse_args()
set_seed(args.random_seed)

features, labels_n, labels, adj, num_nodes, num_feats, label_map = load_citeseer(0)

adj = torch.FloatTensor(adj)
features = torch.FloatTensor(features)
labels_n = torch.LongTensor(np.where(labels)[1])  # 由one-hot编码变为自然数
labels_initial = labels_n
labels = torch.LongTensor(labels)

idx_train, idx_val, idx_test = load_index(
        N=adj.shape[0],
        valid_ratio=0.2,
    )  # 集


idx_clean = torch.LongTensor(range(0,3312))

idx = torch.LongTensor(range(0, 3312 + args.injection_num))

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


Kt_sample = int(args.alpha_cluster*features.shape[1])
Kf_sample = int(args.alpha_cluster*features.shape[1])

gat_model = GAT_Model(
        n_hid=args.hidden_layers,
        in_features=features.shape[-1],
        out_classes=int(labels_n.max()) + 1,
        device=args.device,
        dropout=args.dropout_gat,
        alpha=args.alpha,
        n_head=args.attention_heads,
).to(args.device)  # 模型

gcn_model = GCN(
        nfeat=features.shape[-1], 
        nhid=args.hidden_layers_gcn, 
        nclass=int(labels_n.max()) + 1, 
        dropout=args.dropout_gcn
    ).to(args.device)  # 模型

gcn_model.load_state_dict(torch.load(r"./code/JESTIE/BLA/BLA/pygcn_print_cite_model.ckpt"))
gcn_model.eval()

gcn_output = gcn_model(features.to(args.device), adj.to(args.device)).cpu()
preds_gcn_n = gcn_output.max(-1)[1].type_as(labels_n)
preds_gcn = np.eye(labels.size(-1))[preds_gcn_n]

idx_vic = get_vic_idx(preds_gcn_n, args.victim_class, args.victim_num)
idx_inj = get_inj_idx(args.injection_num, 3312)
adj, features, labels_n, labels = attack_citeseer_data(adj, features, preds_gcn_n, args.injection_num, args.target_class, idx_inj, args)



model = BayesianLinearRegression(idx_train, args, nclass=6) # idx_train[0:700]
model.fit(adj, features, labels, idx_train)
output = model.forward(adj, features, idx_test)


surrogate_output = model.forward(adj, features, idx_clean)
gcn_accuracy_test = accuracy(gcn_output[idx_test], labels_initial[idx_test])
gcn_accuracy = accuracy(gcn_output, labels_initial)
surrogate_accuracy_test = accuracy(surrogate_output[idx_test], labels_initial[idx_test])
surrogate_accuracy = accuracy(surrogate_output, labels_initial)



output_vic = model.forward(adj, features, idx_vic)
labels_vic_values, labels_n_vic = torch.max(output_vic,dim=1)
labels_vic = np.eye(labels.size(-1))[labels_n_vic]
second_class_value, vic_second_class = torch.max((output_vic - 1000*labels_vic),dim=1)


begin_time = time.time()

gamma_vic = compute_feature_importance(features, labels_n, labels.size(-1), features.size(-1))
sampled_features_vic = sample_features(gamma_vic, 200)
adj, features, features_AV_vic, vic_second_class = compute_AV_dict(model, vic_second_class, adj, features, labels_n, idx_inj, idx_vic, sampled_features_vic, args)



features = att_inj_features(model, vic_second_class, adj, features, labels_n, idx_inj, idx_vic, features.shape[-1], args)


adj = edge_attacks(model, adj, features, vic_second_class, idx_inj, idx_vic, 0.1, 3312, 55, args)


end_time = time.time()

gcn_output = gcn_model(features.to(args.device), adj.to(args.device)).cpu()
gcn_acc_vic = 1-accuracy(gcn_output[idx_vic], labels_n[idx_vic])
print("time:{}".format(end_time - begin_time))


