import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.utils import degree, to_dense_adj
from layers import SAGPool, TopKPooling, GSAPool


class Net_SAG(torch.nn.Module):
    def __init__(self,args):
        super(Net_SAG, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.cus_drop_ratio = args.cus_drop_ratio


        # Encoder
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = SAGPool(self.nhid,
                             ratio=self.pooling_ratio,
                             cus_drop_ratio = self.cus_drop_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio,
                             cus_drop_ratio = self.cus_drop_ratio)

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)


        # Feature Decoder
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.conv4 = GCNConv(self.nhid, self.nhid)
        self.conv5 = GCNConv(self.nhid, self.num_features)

        # degree decoder
        self.lin4 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin5 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin6 = torch.nn.Linear(self.nhid//2, 1)



    def forward(self, data):
        x, edge_index_0, batch = data.x, data.edge_index, data.batch

        degree_ground_truth = degree(edge_index_0[0], num_nodes=x.size(0) ).to(x.device)

        x = F.relu(self.conv1(x, edge_index_0))
        res = x
        x, edge_index, _, batch, perm_1, x_ae1 = self.pool1(x, edge_index_0, None, batch)

        # Feature Decoder
        x_out = torch.zeros_like(res)
        x_out[perm_1] = x

        x_decoder = torch.tanh(self.conv3(x_out, edge_index_0))
        x_decoder = torch.tanh(self.conv4(x_decoder, edge_index_0))
        x_decoder_1 = self.conv5(x_decoder, edge_index_0)

        # Degree Decoder
        x_degree = F.relu(self.lin4(x_out))
        x_degree = F.dropout(x_degree, p=self.dropout_ratio, training=self.training)
        x_degree = F.relu(self.lin5(x_degree))
        x_degree = F.dropout(x_degree, p=self.dropout_ratio, training=self.training)
        x_degree = F.relu(self.lin6(x_degree))

        degree_ground_truth_1 =  degree_ground_truth[perm_1]
        degree_predict_1 =  x_degree[perm_1]

        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        res_2 = x
        x, edge_index, _, batch, perm_2, x_ae2 = self.pool2(x, edge_index, None, batch)

        x_out = torch.zeros_like(res_2)
        x_out[perm_2] = x


        x_out_2 = torch.zeros_like(res)
        x_out_2[perm_1] = x_out

        x_decoder = torch.tanh(self.conv3(x_out_2, edge_index_0))
        x_decoder = torch.tanh(self.conv4(x_decoder, edge_index_0))
        x_decoder_2 = self.conv5(x_decoder, edge_index_0)


        x_degree_2 = F.relu(self.lin4(x_out_2))
        x_degree_2 = F.dropout(x_degree_2, p=self.dropout_ratio, training=self.training)
        x_degree_2 = F.relu(self.lin5(x_degree_2))
        x_degree_2 = F.dropout(x_degree_2, p=self.dropout_ratio, training=self.training)
        x_degree_2 = F.relu(self.lin6(x_degree_2))

        degree_ground_truth_2 =  degree_ground_truth[perm_2]
        degree_predict_2 =  x_degree[perm_2]

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)


        x = x1 + x2

        x = F.relu(self.lin1(x))
        #x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x, x_decoder_1, x_decoder_2, degree_ground_truth_1, degree_predict_1, degree_ground_truth_2, degree_predict_2


class Net_GSA(torch.nn.Module):
    def __init__(self,args):
        super(Net_GSA, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.alpha = args.GSA_alpha

        self.pooling_layer_type = args.pooling_layer_type
        self.feature_fusion_type = args.feature_fusion_type

        self.cus_drop_ratio = args.cus_drop_ratio

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = GSAPool(self.nhid,
                            pooling_ratio=self.pooling_ratio,
                            alpha = self.alpha,
                            pooling_conv=self.pooling_layer_type,
                            fusion_conv=self.feature_fusion_type,
                            cus_drop_ratio = self.cus_drop_ratio
                            )
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = GSAPool(self.nhid,
                            pooling_ratio=self.pooling_ratio,
                            alpha = self.alpha,
                             pooling_conv=self.pooling_layer_type,
                             fusion_conv=self.feature_fusion_type,
                             cus_drop_ratio = self.cus_drop_ratio
                             )


        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)


        # Decoder
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.conv4 = GCNConv(self.nhid, self.nhid)
        self.conv5 = GCNConv(self.nhid, self.num_features)


        # degree decoder

        self.conv6 = GCNConv(self.nhid, self.nhid)
        self.conv7 = GCNConv(self.nhid, self.nhid)

        self.lin4 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin5 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin6 = torch.nn.Linear(self.nhid//2, 1)

    def forward(self, data):
        x, edge_index_0, batch = data.x, data.edge_index, data.batch

        degree_ground_truth = degree(edge_index_0[0], num_nodes=x.size(0)).to(x.device)

        x = F.relu(self.conv1(x, edge_index_0))
        res = x
        x, edge_index, _, batch, perm_1, x_ae1 = self.pool1(x, edge_index_0, None, batch)



        # Decoder 1
        x_out = torch.zeros_like(res)
        x_out[perm_1] = x

        x_decoder = torch.tanh(self.conv3(x_out, edge_index_0))
        x_decoder = torch.tanh(self.conv4(x_decoder, edge_index_0))
        x_decoder_1 = self.conv5(x_decoder, edge_index_0)

        x_degree = F.relu(self.lin4(x_out))
        x_degree = F.dropout(x_degree, p=self.dropout_ratio, training=self.training)
        x_degree = F.relu(self.lin5(x_degree))
        x_degree = F.dropout(x_degree, p=self.dropout_ratio, training=self.training)
        x_degree = F.relu(self.lin6(x_degree))

        degree_ground_truth_1 =  degree_ground_truth[perm_1]
        degree_predict_1 =  x_degree[perm_1]

        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        res_2 = x
        x, edge_index, _, batch, perm_2, x_ae2 = self.pool2(x, edge_index, None, batch)

        x_out = torch.zeros_like(res_2)
        x_out[perm_2] = x

        x_out_2 = torch.zeros_like(res)
        x_out_2[perm_1] = x_out

        x_decoder = torch.tanh(self.conv3(x_out_2, edge_index_0))
        x_decoder = torch.tanh(self.conv4(x_decoder, edge_index_0))
        x_decoder_2 = self.conv5(x_decoder, edge_index_0)

        x_degree_2 = F.relu(self.lin4(x_out_2))
        x_degree_2 = F.dropout(x_degree_2, p=self.dropout_ratio, training=self.training)
        x_degree_2 = F.relu(self.lin5(x_degree_2))
        x_degree_2 = F.dropout(x_degree_2, p=self.dropout_ratio, training=self.training)
        x_degree_2 = F.relu(self.lin6(x_degree_2))

        degree_ground_truth_2 =  degree_ground_truth[perm_2]
        degree_predict_2 =  x_degree[perm_2]

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2

        x = F.relu(self.lin1(x))
        #x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x, x_decoder_1, x_decoder_2, degree_ground_truth_1, degree_predict_1, degree_ground_truth_2, degree_predict_2


class Net_TopK(torch.nn.Module):
    def __init__(self, args):
        super(Net_TopK, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.cus_drop_ratio = args.cus_drop_ratio


        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = TopKPooling(self.nhid,
                                 ratio=self.pooling_ratio,
                                 cus_drop_ratio = self.cus_drop_ratio
                                        )
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = TopKPooling(self.nhid,
                                ratio=self.pooling_ratio,
                                cus_drop_ratio = self.cus_drop_ratio
                                        )

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)


        # Decoder
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.conv4 = GCNConv(self.nhid, self.nhid)
        self.conv5 = GCNConv(self.nhid, self.num_features)


        # degree decoder

        self.lin4 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin5 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin6 = torch.nn.Linear(self.nhid//2, 1)

    def forward(self, data):
        x, edge_index_0, batch = data.x, data.edge_index, data.batch

        degree_ground_truth = degree(edge_index_0[0], num_nodes=x.size(0)).to(x.device)


        x = F.relu(self.conv1(x, edge_index_0))
        res = x
        x, edge_index, _, batch, perm_1, x_ae1 = self.pool1(x, edge_index_0, None, batch)

        # Decoder 1
        x_out = torch.zeros_like(res)
        x_out[perm_1] = x

        x_decoder = torch.tanh(self.conv3(x_out, edge_index_0))
        x_decoder = torch.tanh(self.conv4(x_decoder, edge_index_0))
        x_decoder_1 = self.conv5(x_decoder, edge_index_0)

        x_degree = F.relu(self.lin4(x_out))
        x_degree = F.dropout(x_degree, p=self.dropout_ratio, training=self.training)
        x_degree = F.relu(self.lin5(x_degree))
        x_degree = F.dropout(x_degree, p=self.dropout_ratio, training=self.training)
        x_degree = F.relu(self.lin6(x_degree))

        degree_ground_truth_1 =  degree_ground_truth[perm_1]
        degree_predict_1 =  x_degree[perm_1]

        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        res_2 = x
        x, edge_index, _, batch, perm_2, x_ae2 = self.pool2(x, edge_index, None, batch)

        x_out = torch.zeros_like(res_2)
        x_out[perm_2] = x


        x_out_2 = torch.zeros_like(res)
        x_out_2[perm_1] = x_out

        x_decoder = torch.tanh(self.conv3(x_out_2, edge_index_0))
        x_decoder = torch.tanh(self.conv4(x_decoder, edge_index_0))
        x_decoder_2 = self.conv5(x_decoder, edge_index_0)

        x_degree_2 = F.relu(self.lin4(x_out_2))
        x_degree_2 = F.dropout(x_degree_2, p=self.dropout_ratio, training=self.training)
        x_degree_2 = F.relu(self.lin5(x_degree_2))
        x_degree_2 = F.dropout(x_degree_2, p=self.dropout_ratio, training=self.training)
        x_degree_2 = F.relu(self.lin6(x_degree_2))

        degree_ground_truth_2 =  degree_ground_truth[perm_2]
        degree_predict_2 =  x_degree[perm_2]

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 #+ x3

        x = F.relu(self.lin1(x))
        #x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x, x_decoder_1, x_decoder_2, degree_ground_truth_1, degree_predict_1, degree_ground_truth_2, degree_predict_2


    