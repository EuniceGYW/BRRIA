import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans









def attack_citeseer_data(adj, features, labels, injection_num, target_class, idx_inj, args): 
    
    injection_labels = torch.full((injection_num,),target_class)
    injection_features = torch.zeros([injection_num, features.size(-1)])
    idx = np.arange(len(labels))
    for i in range(len(injection_labels)):

        idx_i = idx[labels == injection_labels[i]]
        features_i = features[idx_i]
        features_i_draw = torch.randint(0, features_i.size(0),(1,))  
        injection_features[i] = features[features_i_draw]

    


    adj = torch.cat((adj, torch.zeros([injection_num, adj.size(-1)])), dim = 0)
    adj = torch.cat((adj, torch.zeros([adj.size(0), injection_num])), dim = 1)
    inject_adj_idx = torch.randint(0, 3312, (injection_num, 3))
    
    with open(args.txt_attack_path, "a") as f:
            f.write(
            f"initial adj num of injected nodes:{2:.1f}. \n "
            )
    for item in idx_inj:
        idx_i = idx[labels == target_class]
        inject_adj_idx = torch.randint(0, len(idx_i), (1,3))
       
        for j in inject_adj_idx:
            adj_idx = idx_i[j]
            adj[item, adj_idx] = 1 - adj[item, adj_idx]
            adj[adj_idx, item] = 1 - adj[adj_idx, item]

    


    features = torch.cat((features, injection_features), dim = 0)    
    labels = torch.cat((labels, injection_labels), dim = 0)  
    
 


    labels_inject = torch.tensor(np.eye(6)[labels])

    return adj, features, labels, labels_inject



def get_vic_idx(labels_n, victim_class, victim_num):
    victim_num = int(victim_num)
    idx = np.arange(len(labels_n))

    idx_vic = idx[labels_n == victim_class]
    idx_vic  = np.random.permutation(idx_vic)
    idx_vic = idx_vic[0:victim_num] 

    return torch.LongTensor(idx_vic)





def feature_statistic(features, args):
    feature_num = torch.sum(features, dim=0)
    fea_set_num = len(feature_num)//args.mini_batch 
    feature_beta = torch.zeros((fea_set_num,))

    for i in range(fea_set_num):
        if i < fea_set_num:
            feature_beta[i] = (feature_num[i*args.mini_batch:(i+1)*args.mini_batch]).sum()
        else:
            feature_beta[i] = (feature_num[i*args.mini_batch:]).sum()
    
    return feature_beta


def get_inj_idx(injection_num, num_nodes):

    idx_inj = np.array(range(num_nodes, num_nodes + injection_num))

    return torch.LongTensor(idx_inj)


def loss_function(surrogate_model, vic_second_class, adj, features, idx_inj, idx_vic, alpha, args):
    output_inj = surrogate_model.forward(adj, features, idx_inj)

    loss_inj = output_inj[:,args.target_class].sum()
    output_vic = surrogate_model.forward(adj, features, idx_vic)
    loss_vic = 0.1*(output_vic[:,vic_second_class].sum() - output_vic[:,args.victim_class].sum())
    print("loss_inj:{}, loss_vic:{}.".format(loss_inj, loss_vic))
    loss = alpha * loss_inj + (1-alpha) * loss_vic
    

    return loss

def loss_function_compute(surrogate_model, vic_second_class, adj, features, idx_inj, idx_vic, alpha, args):
    output_inj = surrogate_model.forward(adj, features, idx_inj)

    loss_inj = output_inj[:,args.target_class].sum()
    output_vic = surrogate_model.forward(adj, features, idx_vic)
    loss_vic = 0.1*(output_vic[:,vic_second_class].sum() - output_vic[:,args.victim_class].sum())
    print("loss_inj:{}, loss_vic:{}.".format(loss_inj, loss_vic))
    


    loss = alpha * loss_inj + (1-alpha) * loss_vic
    
    modulation = 0
    if loss_vic == 0.0 and modulation < 1:
            modulation = modulation + 1
            if vic_second_class < 3:
                vic_second_class = vic_second_class + 1
            else:
                vic_second_class = 0

    return loss, vic_second_class

def compute_feature_importance(x, y, num_classes, num_features):  
    """  
    计算每个特征在目标类别中的出现频率，作为特征重要性系数。  
    """  
    gamma = torch.zeros(num_features)  
    for l in range(num_classes):  
        mask = (y == l)  
        if mask.sum() > 0:  
            gamma += x[mask].sum(dim=0)  # 统计特征在类别 l 中的出现次数  
    gamma = gamma / gamma.sum()  # 归一化  
    return gamma  

def compute_feature_importance_vic(x, y, vic_class, tar_class, num_features):  
    """  
    计算每个特征在目标类别中的出现频率，作为特征重要性系数。  
    """  
    gamma = torch.zeros(num_features)  
    mask = (y == vic_class)  
    if mask.sum() > 0:  
        gamma = x[mask].sum(dim=0)  # 统计特征在类别 l 中的出现次数  
    mask = (y == tar_class)  
    if mask.sum() > 0:  
        gamma += x[mask].sum(dim=0)
    gamma = gamma / gamma.sum()  # 归一化  
    return gamma  

def sample_features(gamma, num_samples):  
    """  
    根据特征重要性系数 gamma，随机采样需要优化的特征。  
    """  
    probs = gamma #/ gamma.sum()  # 归一化为概率 
    sampled_features = np.random.choice(len(probs), num_samples, p=probs.numpy(), replace=False)  
    #sampled_features = np.argsort(probs.numpy())[-num_samples:][::-1] 
    return sampled_features  





def get_vic_class_idx(labels_n, victim_class):
    # 准备被攻击类别所有节点的序号。
    idx = np.arange(len(labels_n))

    idx_vic = idx[labels_n == victim_class]

    return torch.LongTensor(idx_vic)

def get_vic_class_fea(labels_n, victim_class, features):
    # 准备被攻击类别所有节点的特征矩阵。
    idx_vic = get_vic_class_idx(labels_n, victim_class)
    fea_vic = features[idx_vic,:]

    return fea_vic 


def get_target_class_idx(labels_n, target_class):
    # 准备被攻击节点的目标类别节点的序号。
    idx = np.arange(len(labels_n))

    idx_target = idx[labels_n == target_class]

    return torch.LongTensor(idx_target)

def get_target_class_fea(labels_n, target_class, features):
    # 准备被攻击节点的目标类别节点的特征矩阵。
    idx_target = get_target_class_idx(labels_n, target_class)
    fea_target = features[idx_target,:]

    return fea_target 
def vic_query(labels_n, features, args):

    fea_vic = get_vic_class_fea(labels_n, args.victim_class, features)
    beta_vic = feature_statistic(fea_vic, args)
    fea_target = get_target_class_fea(labels_n, args.target_class, features)
    beta_target = feature_statistic(fea_target, args)
    # query precedence
    query_prec = torch.exp(torch.add(args.theta_vic*torch.reciprocal(-beta_vic), torch.reciprocal(-beta_target)))
    #query_prec = torch.exp(torch.reciprocal(-beta_vic))

    return query_prec


def inj_query(labels_n, features, args):

    fea_vic = get_vic_class_fea(labels_n, args.victim_class, features)
    beta_vic = feature_statistic(fea_vic, args)
    fea_target = get_target_class_fea(labels_n, args.target_class, features)
    beta_target = feature_statistic(fea_target, args)
    # query precedence
    query_prec = torch.exp(torch.add(torch.reciprocal(-beta_vic), args.theta_inj*torch.reciprocal(-beta_target)))

    return query_prec

def compute_AV(model, vic_second_class, adj, features, labels_n, idx_inj, idx_vic, sampled_features_vic, args):

    idx_feats = torch.LongTensor(np.arange(features.shape[-1]))
    idx_inj = idx_inj.unsqueeze(0)
    idx_vic = idx_vic.unsqueeze(0)


    # 将这个idx_vic与idx_inj[0]相连。
    adj[idx_vic, idx_inj] = 1# - adj[idx_vic, idx_inj]
    adj[idx_inj, idx_vic] = 1# - adj[idx_inj, idx_vic]




    min_loss, vic_second_class = loss_function_compute(model, vic_second_class, adj, features, idx_inj, idx_vic, alpha = 0.1, args = args)


    
    for j in sampled_features_vic:
        features[idx_inj, j] = 1 - features[idx_inj, j]
        loss = loss_function(model, vic_second_class, adj, features, idx_inj, idx_vic, alpha = 0.1, args = args)

        if loss > min_loss: 
            features[idx_inj, j] = features[idx_inj, j]
            min_loss = loss

            #loss = loss_function(model, vic_second_class, adj, features, idx_inj, idx_vic, alpha = 0.1, args = args)

                
        else: 
            features[idx_inj, j] = 1 - features[idx_inj, j]
        
 
    # 将这个idx_vic与idx_inj[0]断开。
    adj[idx_vic, idx_inj] = 1 - adj[idx_vic, idx_inj]
    adj[idx_inj, idx_vic] = 1 - adj[idx_inj, idx_vic]


                    
    return features[idx_inj,:], vic_second_class

def compute_AV_dict(model, vic_second_class, adj, features, labels_n, idx_inj, idx_vic, sampled_features_vic, args):

    min_loss = loss_function(model, vic_second_class, adj, features, idx_inj, idx_vic, alpha = 0.1, args = args)

    with open(args.txt_attack_path, "a") as f:
            f.write(
            f"compute_AV_dict alpha:{0.1:.3f}; \n "
            )

    features_AV_vic = torch.zeros(len(idx_vic), features.shape[-1])
    estimator = KMeans(n_clusters=len(idx_inj))
    for i, vic_idx in enumerate(idx_vic):
        features_AV_vic[i,:], vic_second_class[i] = compute_AV(model, vic_second_class[i], adj, features, labels_n, idx_inj[0], idx_vic[i], sampled_features_vic, args)
    estimator.fit(features_AV_vic)
    #print("estimator.fit(AVs):{}".format(estimator.fit(AVs)))
    label_pred = estimator.labels_
    print("label_pred:{}".format(label_pred))

    # 将每个idx_vic与其对应的idx_inj相连。
    for i, vic_idx in enumerate(idx_vic):
        inj_idx = idx_inj[label_pred[i]]
        adj[vic_idx, inj_idx] = 1 - adj[vic_idx, inj_idx]
        adj[inj_idx, vic_idx] = 1 - adj[inj_idx, vic_idx]
        features[idx_inj] = features_AV_vic[i,:]

    final_loss = loss_function(model, vic_second_class, adj, features, idx_inj, idx_vic, alpha = 0.1, args = args)
    print("min_loss_compute_AV:{}, final_loss_compute_AV:{}".format(min_loss, final_loss))
    with open(args.txt_attack_path, "a") as f:
            f.write(
            f"loss before compute AV:{min_loss:.8f}; \n "
            f"loss after compute AV:{final_loss:.8f}.\n "
            )


    return adj, features, features_AV_vic, vic_second_class

