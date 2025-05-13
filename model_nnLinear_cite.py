
import argparse
import numpy as np


from utils import load_index, load_citeseer
from attack_utils_blr import get_vic_class_idx, get_target_class_idx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from utils import accuracy


class BayesianLinearRegression(nn.Module):
    def __init__(self, in_features, out_features, num_layers=2, hidden_dim=500, hid_dim = 40):

        super(BayesianLinearRegression, self).__init__()
        self.num_layers = num_layers

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(in_features, hidden_dim))
        for _ in range(num_layers-1):

            self.fc_layers.append(nn.Linear(hidden_dim, hid_dim))
        

        self.mu = nn.Linear(hid_dim, out_features)
        self.log_sigma = nn.Linear(hid_dim, out_features)
    
    def forward(self, x, adj_matrix):

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = F.relu(x)
        

        x = self.adj_weighted_mean(x, adj_matrix)
        

        mu = self.mu(x)
        log_sigma = self.log_sigma(x)  
        sigma = torch.exp(log_sigma) 
        return mu, sigma
    
    def adj_weighted_mean(self, x, adj_matrix):

        x_weighted = torch.matmul(adj_matrix, x)
        return x_weighted / adj_matrix.sum(dim=1, keepdim=True)









def train(model, idx_train, idx_val, x, adj_matrix, labels, args):

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
       
    idx_train = idx_train.to(args.device)
    idx_val = idx_val.to(args.device)

        
   

    best_acc = 0.0 
    best_epoch = -1  
    no_improvement = 0  

    for epoch in range(args.num_epoch):
        model.train()  
        optimizer.zero_grad()  

        mu, sigma = model(x, adj_matrix)
        train_loss = -torch.mean(
            -0.5 * torch.log(2 * torch.pi * sigma[idx_train] ** 2) - 
            (labels[idx_train] - mu[idx_train]) ** 2 / (2 * sigma[idx_train] ** 2)
        )  # 求训练集的loss
        train_loss.backward()  
        optimizer.step()  

        train_acc = accuracy(mu[idx_train], labels[idx_train]) 

        print(f"[ Train {epoch + 1:3d} / {args.num_epoch} ] loss = {train_loss.data.item():.5f}, "
              f"acc = {train_acc.data.item():.5f}")  
        with open(args.txt_log_path, "a") as f:
            f.write(
                f"[ Train {epoch + 1:3d} / {args.num_epoch} ] loss = {train_loss.data.item():.5f}, "
                f"acc = {train_acc.data.item():.5f}\n"
            )

        model.eval()  
        mu, sigma = model(x, adj_matrix)
        valid_loss = -torch.mean(
            -0.5 * torch.log(2 * torch.pi * sigma[idx_val] ** 2) - 
            (labels[idx_val] - mu[idx_val]) ** 2 / (2 * sigma[idx_val] ** 2)
        )
        valid_acc = accuracy(mu[idx_val], labels[idx_val])

        print(f"[ Valid {epoch + 1:3d} / {args.num_epoch} ] loss = {valid_loss.data.item():.5f}, "
              f"acc = {valid_acc.data.item():.5f}")
        with open(args.txt_log_path, "a") as f:
            f.write(
                f"[ Valid {epoch + 1:3d} / {args.num_epoch} ] loss = {valid_loss.data.item():.5f}, "
                f"acc = {valid_acc.data.item():.5f}"
            )

        if valid_acc.data.item() > best_acc:  
            print(f"Found Best Model, Saving.")
            torch.save(model.state_dict(), args.model_save_path)
            with open(args.txt_log_path, "a") as f:
                f.write(" -> best\n")
            best_acc = valid_acc.data.item()
            best_epoch = epoch + 1  # 最佳参数对应的训练次数加1。
            no_improvement = 0  # 未增长的次数清0。
        else:  # 否则就只换行，而且未增长次数加1
            with open(args.txt_log_path, "a") as f:
                f.write("\n")
            no_improvement += 1

            if no_improvement >= args.patient:  # 未增长速度大于等于规定的超参数，则停止训练。
                print(f"No improvement {args.patient} consecutive epochs, early stopping.")
                print(f"Best model found at epoch no.{best_epoch}, with acc = {best_acc:.5f}")
                with open(args.txt_log_path, "a") as f:
                    f.write(f"Best model found at epoch no.{best_epoch}, with acc = {best_acc:.5f}")

                return
    print(f"Best model found at epoch no.{best_epoch}, with acc = {best_acc:.5f}")
    with open(args.txt_log_path, "a") as f:
        f.write(f"Best model found at epoch no.{best_epoch}, with acc = {best_acc:.5f}")

def test_model(model,idx_test, x, adj_matrix, labels, args):  # 测试模型
    model.eval()


    idx_test = idx_test.to(args.device)
    

    mu, sigma = model(x, adj_matrix)
    test_loss = -torch.mean(
            -0.5 * torch.log(2 * torch.pi * sigma[idx_test] ** 2) - 
            (labels[idx_test] - mu[idx_test]) ** 2 / (2 * sigma[idx_test] ** 2)
        )
    test_acc = accuracy(mu[idx_test], labels[idx_test])

    print(f"[ Test ] loss = {test_loss.data.item():.5f}, "  # 打印测试结果
          f"acc = {test_acc.data.item():.5f}")
    with open(args.txt_log_path, "a") as f:
        f.write(
            f"[ Test ] loss = {test_loss.data.item():.5f}, "
            f"acc = {test_acc.data.item():.5f}\n"
        )
    return mu, sigma, test_loss, test_acc
def accuracy(output, labels):
    labels = labels.max(1)[1]
    preds = output.max(1)[1].type_as(labels)  
    correct = preds.eq(labels).double()  
    correct = correct.sum()  
    return correct / len(labels)
# 使用示例
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
    parser.add_argument("--random_seed", type=int, default=3407)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--patient", type=int, default=100)

    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--hidden_layers", type=int, default=16)
    parser.add_argument('--dropout_gcn', type=float, default=0.5,help='Dropout rate (1 - keep probability).')
    
    parser.add_argument("--txt_log_path", type=str, default=r"./code/JESTIE/BLA/BLA/BLR_cite_print.txt")  # 打印文件训练集和验证集的结果
    parser.add_argument("--model_save_path", type=str, default=r"./code/JESTIE/BLA/BLA/BLR_cite.pth")  # 保存模型的权重
    parser.add_argument("--GPR_model_save_path", type=str, default=r"./code/JESTIE/BLA/BLA/BLR_cite_model.pth")  # 保存模型的权重


    parser.add_argument("--nhid", type=int, default=256)  # 隐藏层输出维度，这个超参数是需要通过尝试调整的。
 
    parser.add_argument("--dropout", type=int, default=0.8)  # 经验/可以试一试?
    parser.add_argument("--alpha", type=int, default=0.2)  # 经验/可以试一试?LeakyReLU的负值部分的斜率。

   
    # attack
    parser.add_argument("--injection_num", type=float, default=10)  # 注入节点数
    parser.add_argument("--query_num", type=float, default=100)  # 询问总次数
    parser.add_argument("--budget", type=float, default=10)  # 更改特征及连边总数
    parser.add_argument("--victim_class", type=float, default=2)  # 更改特征及连边总数
    parser.add_argument("--target_class", type=float, default=5)  # 更改特征及连边总数

    return parser.parse_args()
args = parse_args()
if "default" == args.device:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
set_seed(args.random_seed)

features, labels_n, labels, adj, num_nodes, num_feats, label_map = load_citeseer(0)


args = parse_args()
adj = torch.FloatTensor(adj)
features = torch.FloatTensor(features)

labels_n = torch.LongTensor(labels_n)

idx = torch.LongTensor(range(0, num_nodes))

idx_train = torch.LongTensor(idx[400:])
idx_val = torch.LongTensor(idx[400:])
idx_test = torch.LongTensor(idx[400:])

idx_vic = get_vic_class_idx(labels_n, args.victim_class)
idx_target = get_target_class_idx(labels_n, args.target_class)

idx_train = torch.LongTensor(idx_train)
idx_train = torch.cat((idx_train, idx_vic), dim = 0)
idx_train = torch.unique(torch.cat((idx_train, idx_target), dim = 0))
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
adj = adj.to(args.device)  
features = features.to(args.device)   
labels = labels.to(args.device)
labels_n = labels_n.to(args.device)

model = BayesianLinearRegression(in_features=features.size(-1), out_features=6).to(args.device)  

train(model, idx_train, idx_val, features, adj, labels, args = args)
test_model(model,idx_test, features, adj, labels, args)



