import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import pep_max, a1_max, a2_max, a3_max, b1_max, b2_max, b3_max

class NetTCRGlobalMax(nn.Module):
    def __init__(self, num_filters, embed_dim, dropout_rate):
        super(NetTCRGlobalMax, self).__init__()
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        kernel_sizes = [1, 3, 5, 7, 9]

        self.pep = nn.Parameter(torch.randn(pep_max, embed_dim))
        self.a1 = nn.Parameter(torch.randn(a1_max, embed_dim))
        self.a2 = nn.Parameter(torch.randn(a2_max, embed_dim))
        self.a3 = nn.Parameter(torch.randn(a3_max, embed_dim))
        self.b1 = nn.Parameter(torch.randn(b1_max, embed_dim))
        self.b2 = nn.Parameter(torch.randn(b2_max, embed_dim))
        self.b3 = nn.Parameter(torch.randn(b3_max, embed_dim))

        self.conv1d_1 = nn.ModuleList(nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=1, padding="same") for _ in range(7))
        self.conv1d_3 = nn.ModuleList(nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=3, padding="same") for _ in range(7))
        self.conv1d_5 = nn.ModuleList(nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=5, padding="same") for _ in range(7))
        self.conv1d_7 = nn.ModuleList(nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=7, padding="same") for _ in range(7))
        self.conv1d_9 = nn.ModuleList(nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=9, padding="same") for _ in range(7))

        self.conv_activation = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)  # GlobalMaxPooling1D equivalent
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(num_filters * 35, 1)
        self.dense_activation = nn.Sigmoid()

    def forward(self, x):
        pep, a1, a2, a3, b1, b2, b3 = x

        # Conv1d layers
        pep = [self.conv1d_1[0](pep), self.conv1d_3[0](pep), self.conv1d_5[0](pep), self.conv1d_7[0](pep), self.conv1d_9[0](pep)]
        a1 = [self.conv1d_1[1](a1), self.conv1d_3[1](a1), self.conv1d_5[1](a1), self.conv1d_7[1](a1), self.conv1d_9[1](a1)]
        a2 = [self.conv1d_1[2](a2), self.conv1d_3[2](a2), self.conv1d_5[2](a2), self.conv1d_7[2](a2), self.conv1d_9[2](a2)]
        a3 = [self.conv1d_1[3](a3), self.conv1d_3[3](a3), self.conv1d_5[3](a3), self.conv1d_7[3](a3), self.conv1d_9[3](a3)]
        b1 = [self.conv1d_1[4](b1), self.conv1d_3[4](b1), self.conv1d_5[4](b1), self.conv1d_7[4](b1), self.conv1d_9[4](b1)]
        b2 = [self.conv1d_1[5](b2), self.conv1d_3[5](b2), self.conv1d_5[5](b2), self.conv1d_7[5](b2), self.conv1d_9[5](b2)]
        b3 = [self.conv1d_1[6](b3), self.conv1d_3[6](b3), self.conv1d_5[6](b3), self.conv1d_7[6](b3), self.conv1d_9[6](b3)]

        # Activation functions
        pep = [self.conv_activation(i) for i in pep]
        a1 = [self.conv_activation(i) for i in a1]
        a2 = [self.conv_activation(i) for i in a2]
        a3 = [self.conv_activation(i) for i in a3]
        b1 = [self.conv_activation(i) for i in b1]
        b2 = [self.conv_activation(i) for i in b2]
        b3 = [self.conv_activation(i) for i in b3]

        # pooling layers
        pep = [self.global_max_pool(i) for i in pep]
        a1 = [self.global_max_pool(i) for i in a1]
        a2 = [self.global_max_pool(i) for i in a2]
        a3 = [self.global_max_pool(i) for i in a3]
        b1 = [self.global_max_pool(i) for i in b1]
        b2 = [self.global_max_pool(i) for i in b2]
        b3 = [self.global_max_pool(i) for i in b3]

        # concatentation
        out = torch.cat((pep+a1+a2+a3+b1+b2+b3), dim=1)

        # dropout
        out = self.dropout(out)

        out = out

        # dense layers
        out = torch.squeeze(out)
        out = self.dense(out)

        # sigmoid activation
        out = self.dense_activation(out)
        out = torch.squeeze(out)


        return out
        
class NetTCRGlobalMax2StepsPretraining(nn.Module):
    def __init__(self, num_filters, embed_dim, dropout_rate):
        super(NetTCRGlobalMax2StepsPretraining, self).__init__()
        a1_max, a2_max, a3_max, b1_max, b2_max, b3_max, pep_max = 7, 8, 22, 6, 7, 23, 12, 20
        kernel_sizes = [1, 3, 5, 7, 9]

        self.pep = nn.Parameter(torch.randn(pep_max, embed_dim))
        self.a1 = nn.Parameter(torch.randn(a1_max, embed_dim))
        self.a2 = nn.Parameter(torch.randn(a2_max, embed_dim))
        self.a3 = nn.Parameter(torch.randn(a3_max, embed_dim))
        self.b1 = nn.Parameter(torch.randn(b1_max, embed_dim))
        self.b2 = nn.Parameter(torch.randn(b2_max, embed_dim))
        self.b3 = nn.Parameter(torch.randn(b3_max, embed_dim))

        self.conv1d_1 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=1, padding="same")
        self.conv1d_3 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=3, padding="same")
        self.conv1d_5 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=5, padding="same")
        self.conv1d_7 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=7, padding="same")
        self.conv1d_9 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=9, padding="same")

        self.conv_activation = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)  # GlobalMaxPooling1D equivalent
        self.concat = nn.Concatenate(dim=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(32, 64)  # 32 is the number of output channels from Conv1D * 2
        self.dense_activation = nn.Sigmoid()
        self.out = nn.Linear(64, 1)  # Output layer

    def forward(self, x):
        pep, a1, a2, a3, b1, b2, b3 = x

        # Conv1d layers
        pep = [self.conv1d_1(pep), self.conv1d_3(pep), self.conv1d_5(pep), self.conv1d_7(pep), self.conv1d_9(pep)]
        a1 = [self.conv1d_1(a1), self.conv1d_3(a1), self.conv1d_5(a1), self.conv1d_7(a1), self.conv1d_9(a1)]
        a2 = [self.conv1d_1(a2), self.conv1d_3(a2), self.conv1d_5(a2), self.conv1d_7(a2), self.conv1d_9(a2)]
        a3 = [self.conv1d_1(a3), self.conv1d_3(a3), self.conv1d_5(a3), self.conv1d_7(a3), self.conv1d_9(a3)]
        b1 = [self.conv1d_1(b1), self.conv1d_3(b1), self.conv1d_5(b1), self.conv1d_7(b1), self.conv1d_9(b1)]
        b2 = [self.conv1d_1(b2), self.conv1d_3(b2), self.conv1d_5(b2), self.conv1d_7(b2), self.conv1d_9(b2)]
        b3 = [self.conv1d_1(b3), self.conv1d_3(b3), self.conv1d_5(b3), self.conv1d_7(b3), self.conv1d_9(b3)]

        # Activation functions
        pep = [self.conv_activation(i) for i in pep]
        a1 = [self.conv_activation(i) for i in a1]
        a2 = [self.conv_activation(i) for i in a2]
        a3 = [self.conv_activation(i) for i in a3]
        b1 = [self.conv_activation(i) for i in b1]
        b2 = [self.conv_activation(i) for i in b2]
        b3 = [self.conv_activation(i) for i in b3]

        # pooling layers
        pep = [self.global_max_pool(i) for i in pep]
        a1 = [self.global_max_pool(i) for i in a1]
        a2 = [self.global_max_pool(i) for i in a2]
        a3 = [self.global_max_pool(i) for i in a3]
        b1 = [self.global_max_pool(i) for i in b1]
        b2 = [self.global_max_pool(i) for i in b2]
        b3 = [self.global_max_pool(i) for i in b3]

        # concatentation
        out = self.concat(pep + a1 + a2 + a3 + b1 + b2 + b3)

        # dropout
        out = self.dropout(out)

        # dense layers
        out = self.dense(out)

        # activation
        out = self.dense_activation(out)

        # linear output layer
        out = self.out(out)
        return out
        

