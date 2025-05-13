import numpy as np
import torch


def load_index(N, valid_ratio):  # 返回了训练集、验证集和测试集
    test_index = []  # 制空列表
    for line in open(r"./code/JESTIE/BLA/BLA/dataset/ind.citeseer.test.index"):  # test集的节点序号，一共1000个
        test_index.append(int(line.strip()))  # 这个函数是去掉空格后者换行符
    test_index = np.array(np.sort(test_index))  # 按顺序排列

    train_and_valid_index = np.setdiff1d(  # 取test_index的补集作为训练集和验证集
        range(N),
        test_index,
        assume_unique=True,
    )
    num_train_and_valid = len(train_and_valid_index)  # 训练集和验证集中的节点数量

    np.random.shuffle(train_and_valid_index)  # 打乱数据顺序，方便之后两行随机选取
    train_index = train_and_valid_index[:int(num_train_and_valid * (1 - valid_ratio))]  # valid_ratio这里为0.2。随机选取训练集和验证集中的80%的数据。
    valid_index = train_and_valid_index[len(train_index):]

    train_index = np.sort(train_index)  # 按顺序排列
    valid_index = np.sort(valid_index)

    return train_index, valid_index, test_index



def load_citeseer(num_fake_nodes):
    num_nodes = 3312
    num_feats = 3703
    feat_data = np.zeros((num_nodes + num_fake_nodes, num_feats)) # A big feature matrix
    labels_n = np.empty((num_nodes + num_fake_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open(r"./code/JESTIE/GAB/dataset/citeseer/citeseer.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i # mapping the index of paper into [0, number of papers)
            if not info[-1] in label_map: # info[-1] is the label of paper
                label_map[info[-1]] = len(label_map)
            labels_n[i] = label_map[info[-1]]

    cnt_no = 0
    A = np.zeros((num_nodes,num_nodes))  # 确定邻接矩阵形状
    with open(r"./code/JESTIE/GAB/dataset/citeseer/citeseer.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            if(info[0] not in node_map.keys()) or (info[1] not in node_map.keys()):
                cnt_no += 1
                continue
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            #print("info[0]:{},info[1]:{}".format(info[0],info[1]))
            #print("paper1:{},paper2:{}".format(paper1,paper2))
            A[paper1][paper2] = 1
            A[paper2][paper1] = 1
    A += np.eye(num_nodes)  # 定义节点自身，加上单位阵，使对角元素为1
    print("Totally omit %d dangling edges." %(cnt_no))


    labels_n = torch.squeeze(torch.LongTensor(labels_n), dim = 1)

    labels_inject = torch.LongTensor(np.eye(6)[labels_n.numpy()])
    return feat_data, labels_n, labels_inject, A, num_nodes, num_feats, len(label_map)



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  # 输出中的最大值的索引值（第几行），且与labels类型相同
    correct = preds.eq(labels).double()  # 判断与labels的值是否相同，相同为double型的1，否则为0。
    correct = correct.sum()  # 把所有比较结果加到一起
    return correct / len(labels)  # 除以与labels的长度，即参与判断的节点个数，得到accuracy




