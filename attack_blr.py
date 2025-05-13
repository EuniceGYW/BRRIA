import torch
import torch.nn.functional as F
from attack_utils_blr import vic_query, loss_function
import numpy as np



def att_inj_features(model, vic_second_class, adj, features, labels_n, idx_inj, idx_vic, num_feats, args):
    idx_feats = torch.LongTensor(np.arange(num_feats))
    query_prec = vic_query(labels_n, features, args)
    query_batch, query_batch_idx = torch.sort(query_prec, descending=True)

    min_loss = loss_function(model, vic_second_class, adj, features, idx_inj, idx_vic, alpha = 0.1, args = args)

    for nodes in idx_inj:
        for i in query_batch_idx[:40]:
            idx_feats_sample = torch.LongTensor(np.random.permutation(idx_feats[i*args.mini_batch:(i+1)*args.mini_batch]))
            idx_feats_sample = idx_feats_sample[0:40]
            for j in idx_feats_sample:
                features[nodes, j] = 1 - features[nodes, j]
                loss = loss_function(model, vic_second_class, adj, features, idx_inj, idx_vic, alpha = 0.1, args = args)
                if loss > min_loss: 
                    features[nodes, j] = features[nodes, j]
                    min_loss = loss

                else: 
                    features[nodes, j] = 1 - features[nodes, j]
        
        for i in query_batch_idx[40:80]:
            idx_feats_sample = torch.LongTensor(np.random.permutation(idx_feats[i*args.mini_batch:(i+1)*args.mini_batch]))
            idx_feats_sample = idx_feats_sample[0:40]
            for j in idx_feats_sample:
                features[nodes, j] = 1 - features[nodes, j]
                loss = loss_function(model, vic_second_class, adj, features, idx_inj, idx_vic, alpha = 0.1, args = args)
                if loss > min_loss: 
                    features[nodes, j] = features[nodes, j]
                    min_loss = loss

                else: 
                    features[nodes, j] = 1 - features[nodes, j]

        for i in query_batch_idx[80:]:
            idx_feats_sample = torch.LongTensor(np.random.permutation(idx_feats[i*args.mini_batch:(i+1)*args.mini_batch]))
            idx_feats_sample = idx_feats_sample[0:10]
            for j in idx_feats_sample:
                features[nodes, j] = 1 - features[nodes, j]
                loss = loss_function(model, vic_second_class, adj, features, idx_inj, idx_vic, alpha = 0.1, args = args)
                if loss > min_loss: 
                    features[nodes, j] = features[nodes, j]
                    min_loss = loss

                else: 
                    features[nodes, j] = 1 - features[nodes, j]



    return features


def edge_attacks(model, adj, features, vic_second_class, idx_inj, idx_vic, alpha, num_nodes, init_query, args):
    idx_clean = np.arange(num_nodes)

    idx = np.delete(idx_clean, idx_vic, axis=0)

    min_loss = loss_function(model, vic_second_class, adj, features, idx_inj, idx_vic, alpha = alpha, args = args)


    for inj_idx in idx_inj:
        score_edges = torch.full([num_nodes,],-1000)  # num_nodes为干净的节点数。
        other_edges = torch.randint(1, 3, (1, ))
        idx  = torch.LongTensor(np.random.permutation(idx))
        other_edges_idx = idx[0:init_query]
        for edge_idx in other_edges_idx:
            adj[inj_idx, edge_idx] = 1 - adj[inj_idx, edge_idx]
            adj[edge_idx, inj_idx] = 1 - adj[edge_idx, inj_idx]
            loss = loss_function(model, vic_second_class, adj, features, idx_inj, idx_vic, alpha = alpha, args = args)
            if loss > min_loss: 
                #print("edge_attacks成功")

                score_edges[edge_idx] = loss
                adj[inj_idx, edge_idx] = 1 - adj[inj_idx, edge_idx]
                adj[edge_idx, inj_idx] = 1 - adj[edge_idx, inj_idx]
                min_loss = loss   
            else:
                score_edges[edge_idx] = -1000
                adj[inj_idx, edge_idx] = 1 - adj[inj_idx, edge_idx]
                adj[edge_idx, inj_idx] = 1 - adj[edge_idx, inj_idx]
            values, indices = torch.topk(score_edges, k = other_edges[0])

            for j in indices:
                adj[inj_idx, j] = 1 - adj[inj_idx, j]
                adj[j, inj_idx] = 1 - adj[j, inj_idx]


    return adj






