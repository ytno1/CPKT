import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class Data(Dataset):
    def __init__(self, y, skill, problem, timestamp, real_len):
        super(Data, self).__init__()
        self.y = y
        self.skill = skill
        self.problem = problem
        self.timestamp = timestamp
        self.real_len = real_len

    def __getitem__(self, index):
        return self.y[index], self.skill[index], self.problem[index], self.timestamp[index], self.real_len[index]

    def __len__(self):
        return len(self.problem)


def pad_collate(batch):
    y, skill, problem, timestamp, real_len = zip(*batch)
    item_ids = [torch.LongTensor(i) for i in problem]
    skill_ids = [torch.LongTensor(i) for i in skill]
    timestamp = [torch.LongTensor(i) for i in timestamp]
    labels = [torch.LongTensor(i) for i in y]
    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i))[:-1] for i in item_ids]
    skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

    seq_lists = [item_inputs, skill_inputs, item_ids, skill_ids, timestamp, label_inputs, labels]

    inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0) for seqs in seq_lists[:-2]]
    labels = [pad_sequence(seqs, batch_first=True, padding_value=-1) for seqs in seq_lists[-2:]]  # Pad labels with -1

    return *inputs_and_ids, *labels


def load_dataset(file_path, batch_size, train_ratio=0.7, val_ratio=0.2):
    """Prepare batches grouping padded sequences.

    Arguments:
        file_path: data path
        batch_size (int): number of sequences per batch
        train_ratio,val_ratio: split ratio of Training Valid Test set
    Output:
        batches (list of lists of torch Tensor)
    """
    data = np.load(file_path, allow_pickle=True)
    y, skill, problem, timestamp, real_len = data['y'], data['skill'], data['problem'], data['time'], data['real_len']
    skill_num, pro_num = data['skill_num'], data['problem_num']
    student_num = len(problem)
    print('student_num %d, problem number %d, skill number %d' % (student_num, pro_num, skill_num))

    kt_dataset = Data(y, skill, problem, timestamp, real_len)
    train_size = int(train_ratio * student_num)
    val_size = int(val_ratio * student_num)
    test_size = student_num - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(kt_dataset, [train_size, val_size, test_size])
    print('train_size: ', train_size, 'val_size: ', val_size, 'test_size: ', test_size)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    return pro_num, skill_num, train_data_loader, valid_data_loader, test_data_loader
