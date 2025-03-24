import ast

import torch
import numpy as np
import pandas as pd


def get_corrects(outputs):
    pre_y_list = []
    for o in outputs:
        if o > 0.5:
            pre_y_list.append(1)
        else:
            pre_y_list.append(0)
    return torch.tensor(pre_y_list)


def get_acc(all_true_y, all_pre_y):
    all_pre_y = get_corrects(all_pre_y)
    equal_elements = torch.eq(all_pre_y, all_true_y)
    num_equal = torch.sum(equal_elements.int()).item()
    acc = round(num_equal / all_pre_y.shape[0], 6)
    return acc


def get_features(feature_path, feature_index):
    features_dict = {}
    with open(feature_path, 'r') as f:
        for line in f:
            line = line.strip('\n').split('\t')
            features_dict[int(line[0])] = ast.literal_eval(line[1])
    f.close()
    features_matrix = []
    for i in range(len(features_dict)):
        features_matrix.append(features_dict[i])
    start, end = min(feature_index), max(feature_index) + 1
    features_matrix = torch.tensor(features_matrix)[:, start:end]
    return features_matrix


def get_e2s_map(data, problem_id, skill_id):
    e2s_map = {}
    exercise_group = data.groupby(problem_id)
    max_skill_num = 0
    for exercise_id, group in exercise_group:
        skills = pd.DataFrame(group, dtype=int)[skill_id].unique().tolist()
        e2s_map[exercise_id] = skills
        if len(skills) > max_skill_num:
            max_skill_num = len(skills)
    e2s_map_tensor = torch.zeros(len(e2s_map), max_skill_num)
    for i in range(len(e2s_map)):
        skills_padding = torch.tensor([item + 1 for item in e2s_map[i]] + [0] * (max_skill_num - len(e2s_map[i])))
        e2s_map_tensor[i] = skills_padding
    return e2s_map, e2s_map_tensor


def get_adj(e2s_map, num_exercise, num_skill):
    adj_matrix = np.zeros((num_exercise + num_skill, num_exercise + num_skill), dtype=int)
    adj_map = {key + num_skill: value for key, value in e2s_map.items()}
    # print(adj_map)
    for e, skills in adj_map.items():
        for s in skills:
            adj_matrix[e, s], adj_matrix[s, e] = 1, 1
    return adj_matrix


def get_log(model_log, path):
    model_log.to_excel(path, index=False)


def get_normalized_adj(A):
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return torch.tensor(A_wave).to(torch.float32)
