from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class ExerciseEncoderMLP(nn.Module):
    def __init__(self, input_dim, hidden_size, num_exercise, device):
        super(ExerciseEncoderMLP, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = 1
        self.device = device
        self.lstm = nn.LSTM(input_dim * 2, hidden_size, self.layer_dim, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_exercise)
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
        out, _ = self.lstm(x.to(torch.float32), (h0, c0))
        out = self.dropout(out)
        res = self.linear(out)
        return res


class FeatureEncoder(nn.Module):
    def __init__(self, f_hidden_size, feature_matrix):
        super(FeatureEncoder, self).__init__()
        self.feature_matrix = feature_matrix
        self.exercise_num = feature_matrix.shape[0]
        self.linear_f = nn.Linear(1, f_hidden_size)
        self.linear_d = nn.Linear(1, f_hidden_size)
        self.linear_c = nn.Linear(1, f_hidden_size)
        self.linear_all = nn.Linear(f_hidden_size, f_hidden_size)
        self.linear_influence = nn.Linear(f_hidden_size, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_feature = self.feature_matrix[x.to(torch.long)]
        x_feature_f = self.linear_f(x_feature[:, :, 0].unsqueeze(-1)).unsqueeze(-1)
        x_feature_d = self.linear_d(x_feature[:, :, 1].unsqueeze(-1)).unsqueeze(-1)
        x_feature_c = self.linear_c(x_feature[:, :, 2].unsqueeze(-1)).unsqueeze(-1)
        x_feature_input = torch.concat([x_feature_f, x_feature_d, x_feature_c], dim=-1)
        x_feature_input = self.conv(x_feature_input.permute(0, 3, 1, 2)).squeeze()
        x_feature_input = self.linear_all(x_feature_input)
        influence = self.tanh(self.linear_influence(x_feature_input))
        return x_feature_input, influence


class GCNLayer(nn.Module):
    def __init__(self, adj_hat, num_skill, num_exercise, hidden_size):
        super(GCNLayer, self).__init__()
        self.adj_hat = adj_hat
        self.num_skill = num_skill
        self.num_nodes = num_skill + num_exercise
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(num_skill, hidden_size, 1, True)
        self.linear_att = nn.Linear(self.num_nodes, self.num_nodes)
        self.linear_out = nn.Linear(hidden_size, num_skill)

        self.block_1 = MLBlock(adj_hat, num_skill, num_exercise, hidden_size)
        self.block_2 = MLBlock(adj_hat, num_skill, num_exercise, hidden_size)
        self.block_3 = MLBlock(adj_hat, num_skill, num_exercise, hidden_size)

    def forward(self, all_state_init, influence):
        all_state_update_1 = self.block_1(all_state_init, influence)
        all_state_update_2 = self.block_2(all_state_update_1, influence)
        all_state_update_3 = self.block_3(all_state_update_2, influence)
        all_state_update = all_state_update_1 + all_state_update_3

        return all_state_update


class MLBlock(nn.Module):
    def __init__(self, adj_hat, num_skill, num_exercise, hidden_size):
        super(MLBlock, self).__init__()
        self.num_skill = num_skill
        self.num_nodes = num_skill + num_exercise
        self.relu = nn.ReLU()
        self.adj_hat = adj_hat

        self.lstm = nn.GRU(num_skill, hidden_size, 1, True)
        self.linear_att = nn.Linear(self.num_nodes, self.num_nodes)
        self.linear_out = nn.Linear(hidden_size, num_skill)

    def forward(self, all_state_init, influence):
        exercise_state = all_state_init[:, :, self.num_skill:]
        skill_state = all_state_init[:, :, 0:self.num_skill]
        exercise_state = exercise_state * (1. + influence)
        all_state_init_influence = torch.cat([skill_state, exercise_state], dim=-1)
        all_state = torch.einsum("ij,jkl->lki", [self.adj_hat, all_state_init_influence.permute(2, 1, 0)])
        all_state = self.relu(self.linear_att(all_state * F.softmax(all_state, dim=-1)))
        skill_state = all_state[:, :, 0:self.num_skill]
        skill_state = self.linear_out(self.lstm(skill_state)[0])
        all_state_update = torch.cat((skill_state, all_state[:, :, self.num_skill:]), dim=-1)
        return all_state_update
