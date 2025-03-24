import random
import pandas as pd
import os
import numpy as np
import warnings
from my_dataset import CustomDataset
import torch

warnings.filterwarnings('ignore')


def process_csv(data_path, ori_data='skill_builder_data_corrected.csv',
                out_data='assist09_processed.csv', min_response_num=3, min_exercise_num=10):
    data = pd.read_csv(os.path.join(data_path, ori_data), encoding='ISO-8859-1',
                       low_memory=False).dropna(subset=['skill_id'])
    data = re_id(data)
    ori_problem_num = len(data['problem_id'].unique().tolist())
    ori_skill_num = len(data['skill_id'].unique().tolist())
    print('[Original] Problem number is: {}, skill number is: {}.'
          .format(ori_problem_num, ori_skill_num))
    data = data[~data['skill_id'].isin(['noskill'])]
    data = data[data['original'].isin([1])]
    exercises = data.groupby(['problem_id'], as_index=True)
    delete_exercises = []
    for e in exercises:
        if len(e[1]) < min_exercise_num:
            # delete_exercises.append(e[0][0])
            delete_exercises.append(e[0][0])
    data = data[~data['problem_id'].isin(delete_exercises)]
    print('Delete exercises number based on min_exercise_num is: {}.'.format(len(delete_exercises)))
    students = data.groupby(['user_id'], as_index=True)
    delete_students = []
    for s in students:
        if len(s[1]) < min_response_num:
            delete_students.append(s[0][0])
    print('Delete students number based on min_inter_num is: {}.'.format(len(delete_students)))
    data = data[~data['user_id'].isin(delete_students)]
    data = re_id(data)
    data.to_csv(os.path.join(data_path, out_data), index=False)
    problem_num = len(data['problem_id'].unique().tolist())
    skill_num = len(data['skill_id'].unique().tolist())
    print('[Processed] Problem number is: {}, skill number is: {}.'
          .format(problem_num, skill_num))
    print('Original records processed.')


def re_id(data):
    data['problem_id'] = pd.factorize(data['problem_id'])[0]
    data['skill_id'] = pd.factorize(data['skill_id'])[0]
    return data


def split_by_student(data):
    students = data['user_id'].unique()
    all_sequences = []
    for student_id in students:
        student_seq = data[data['user_id'] == student_id].sort_values('order_id')
        q = student_seq['problem_id'].tolist()
        a = student_seq['correct'].tolist()
        all_sequences.append([q, a])
    return all_sequences


def write_sequence(sequences, path):
    with open(path, 'a', encoding='utf8') as f:
        for seq in sequences:
            questions, answers = seq
            seq_len = len(questions)
            f.write(str(seq_len) + '\n')
            f.write(','.join([str(q) for q in questions]) + '\n')
            f.write(','.join([str(a) for a in answers]) + '\n')
    f.close()


def train_test_split(data, split_rate=.8, shuffle=True):
    if shuffle:
        random.shuffle(data)
    x_seqs_size = len(data)
    num_train = int(x_seqs_size * split_rate)
    train_data = data[: num_train]
    test_data = data[num_train:]
    return train_data, test_data


def encode_input(sequences, max_step):
    q_list, a_list = [], []
    for q, a in sequences:
        mod = -(-len(q) // max_step)
        for i in range(mod):
            start = i * max_step
            end = (i + 1) * max_step
            q_subarray = q[start:end]
            a_subarray = a[start:end]
            if len(q_subarray) < max_step:
                q_subarray = np.pad(q_subarray, (0, max_step - len(q_subarray)), constant_values=0)
                a_subarray = np.pad(a_subarray, (0, max_step - len(a_subarray)), constant_values=-1)
            q_list.append(q_subarray)
            a_list.append(a_subarray)
    return np.array(q_list), np.array(a_list)


def get_exercise_feature(data):
    s_difficulty, e_difficulty = D_encoder(data)
    e_frequency = F_encoder(data)
    k_count = K_encoder(data)
    exercises = data['problem_id'].unique().tolist()
    nodes_features = {}
    for e in exercises:
        nodes_features[e] = [e_difficulty[e], e_frequency[e], k_count[e]]
    return nodes_features


def F_encoder(data):
    data = data[['problem_id', 'skill_id']]
    problem_id_counts = data['problem_id'].value_counts()
    max_count = float(pd.DataFrame(problem_id_counts)['count'].max())
    problem_id_counts = round(problem_id_counts / max_count, 6)
    e_frequency = problem_id_counts.to_dict()
    return e_frequency


def D_encoder(data):
    data = data[['user_id', 'problem_id', 'skill_id', 'correct']]
    s_difficulty, e_difficulty = {}, {}
    skill_group = data.groupby('skill_id')
    exercise_group = data.groupby('problem_id')

    for skill, group in skill_group:
        if len(group) > 30:
            s_correct_count = (group['correct'] == 1).sum()
            s_difficulty[skill] = round(s_correct_count / len(group), 6)
        else:
            s_difficulty[skill] = 0.01
    for exercise, group in exercise_group:
        e_correct_count = (group['correct'] == 1).sum()
        e_difficulty[exercise] = round(e_correct_count / len(group), 6)
    return s_difficulty, e_difficulty


def K_encoder(data):
    k_dict = {}
    exercise_group = data[['skill_id', 'problem_id']].groupby('problem_id')
    for exercise, group in exercise_group:
        kc_num = len(pd.DataFrame(group).value_counts())
        k_dict[exercise] = kc_num
    max_num = max(k_dict.values())
    for key in k_dict:
        k_dict[key] = round(k_dict[key] / max_num, 6)
    return k_dict


def write_dict(out_path, dict_value):
    with open(out_path, 'w') as f:
        for key, value in dict_value.items():
            f.write(f'{key}\t{value}\n')
    f.close()


if __name__ == '__main__':
    data_path = ''
    max_len = 100
    original_file_name = 'skill_builder_data_corrected.csv'
    processed_file_name = 'assist09_processed.csv'
    if not os.path.exists(os.path.join(data_path, processed_file_name)):
        process_csv(data_path, original_file_name, processed_file_name, 3, 10)
    else:
        print('The original data is already processed.')
    df_processed = pd.read_csv(os.path.join(data_path, processed_file_name), encoding='ISO-8859-1',
                               low_memory=False).dropna(subset=['skill_id'])

    sequences = split_by_student(df_processed)
    print("Input sequence count is: {}.".format(len(sequences)))
    # write_sequence(sequences, os.path.join(data_path, 'all_sequences.txt'))

    train_sequences, test_sequences = train_test_split(sequences, 0.8)
    # write_sequence(train_sequences, os.path.join(data_path, 'train_sequences.txt'))
    # write_sequence(test_sequences, os.path.join(data_path, 'test_sequences.txt'))

    train_q, train_a = encode_input(train_sequences, max_len)
    print('Train data shape is: {}.'.format(train_q.shape))
    test_q, test_a = encode_input(test_sequences, max_len)
    print('Test data shape is: {}.'.format(test_q.shape))

    train_dataset, test_dataset = (CustomDataset(train_q, train_a),
                                   CustomDataset(test_q, test_a))

    torch.save(train_dataset, os.path.join(data_path, 'train_dataset.pth'))
    torch.save(test_dataset, os.path.join(data_path, 'test_dataset.pth'))
    print('Train and test dataset save done.')

    processed_data = pd.read_csv(processed_file_name)
    problem_features = get_exercise_feature(processed_data)
    problem_num = len(processed_data['problem_id'].unique().tolist())
    if problem_num == len(problem_features):
        write_dict('problem_features.txt', problem_features)
        print('Exercise number is {}, and feature saved.'.format(problem_num))
    else:
        print('Exercise number is wrong.')
