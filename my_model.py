import torch
import torch.nn as nn
from layers import GCNLayer, ExerciseEncoderMLP, FeatureEncoder
import torch.nn.init as init


class IPDGKT(nn.Module):
    def __init__(self,
                 e_hidden_size,
                 s_hidden_size,
                 f_hidden_size,
                 num_exercise,
                 num_skill,
                 feature_matrix,
                 adj,
                 e2s_map,
                 device):
        super(IPDGKT, self).__init__()

        self.device = device
        self.num_skill = num_skill
        self.num_exercise = num_exercise
        self.adj = adj.to(device)
        self.e2s_map = e2s_map.to(device)

        self.e_encoder = ExerciseEncoderMLP(f_hidden_size, e_hidden_size, self.num_exercise, self.device)
        self.f_encoder = FeatureEncoder(f_hidden_size, feature_matrix)
        self.predict = nn.Linear(num_skill + num_exercise, num_skill)
        self.gcn = GCNLayer(self.adj, num_skill, num_exercise, s_hidden_size)
        self.epsilon = 1e-8
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

    def forward(self, x, y):
        feature_input, influence = self.f_encoder(x)
        inputs = self.encode_inputs(feature_input, y).to(self.device)
        exercise_state = self.e_encoder(inputs)
        all_state = self.trans_state_gcn(exercise_state, influence)
        skill_state = self.predict(all_state)
        e_state = torch.gather(exercise_state, -1, x.to(torch.int64).unsqueeze(-1))
        s_state = self.get_skill_state(skill_state, x.to(torch.int64))
        return torch.sigmoid(e_state), torch.sigmoid(s_state)

    def trans_state_gcn(self, exercise_state, influence):
        batch_len, max_len = exercise_state.shape[0], exercise_state.shape[1]
        skill_state_init = torch.randn(batch_len, max_len, self.num_skill).to(self.device)
        all_state_init = torch.cat((skill_state_init, exercise_state), -1)
        all_state = self.gcn(all_state_init, influence)
        return all_state

    def encode_inputs(self, data_x, data_y):
        data_y = data_y.unsqueeze(-1)
        inputs_r = data_x * data_y
        inputs_w = -(data_x * (data_y - 1))
        inputs = torch.cat([inputs_r, inputs_w], dim=-1)
        return inputs

    def get_skill_state(self, skill_state, x):
        zero_padding = torch.zeros(skill_state.shape[0], skill_state.shape[1], 1).to(self.device)
        skill_state_temp = torch.cat((zero_padding, skill_state), dim=-1)
        skill_state_out = torch.gather(skill_state_temp, -1, self.e2s_map[x].to(torch.int64))
        non_zero = torch.count_nonzero(self.e2s_map[x], dim=-1).unsqueeze(-1)
        skill_state_out = (torch.sum(skill_state_out, dim=-1).unsqueeze(-1) / non_zero) + self.epsilon
        return skill_state_out
