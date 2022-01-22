import argparse
import time as t
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from model_ykt import YKT
from utils import *

"""Code reuse from https://github.com/shalini1194/RKT (https://arxiv.org/abs/2008.12736)
"""

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


def compute_auc(preds, labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    if len(torch.unique(labels)) == 1:  # Only one class
        auc = accuracy_score(labels, preds.round())
        acc = auc
    else:
        auc = roc_auc_score(labels, preds)
        acc = accuracy_score(labels, preds.round())
    return auc, acc


def compute_loss(preds, labels, criterion):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    return criterion(preds, labels)


def computeRePos(time_seq, time_span):
    batch_size = time_seq.shape[0]
    size = time_seq.shape[1]

    time_matrix = (torch.abs(torch.unsqueeze(time_seq, axis=1).repeat(1, size, 1).reshape((batch_size, size * size, 1))
                             - torch.unsqueeze(time_seq, axis=-1).repeat(1, 1, size, ).reshape(
        (batch_size, size * size, 1))))

    time_matrix = time_matrix.reshape((batch_size, size, size))

    return (time_matrix)


def get_corr_data(dataset, pro_num):
    pro_pro_dense = np.zeros((pro_num+1, pro_num+1))
    data_path = './data/' + dataset + '_corr'
    pro_pro_ = open(data_path)
    for i in pro_pro_:
        j = i.strip().split(',')
        pro_pro_dense[int(j[0])][int(j[1])] += int(float(j[2]))
    return pro_pro_dense


def computeREN(skill_seq):
    """Calculate exercising counts matrix
    """
    size = skill_seq.shape[1]
    target = (skill_seq.unsqueeze(1).repeat(1, size, 1) == skill_seq.unsqueeze(-1).repeat(1, 1, size)).float()
    ren_matrix = torch.cumsum(target, dim=-1) * target
    return ren_matrix


def train(train_data, valid_data, timespan, model, corr_data, optimizer, logger, saver, num_epochs, grad_clip):
    """Train YKT model.
    Arguments:
        train_data (list of tuples of torch Tensor)
        val_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        saver: wrapper for torch saving
        num_epochs (int): number of epochs to train for
        batch_size (int)
        grad_clip (float): max norm of the gradients
    """
    criterion = nn.BCEWithLogitsLoss()

    step = 0
    metrics = Metrics()
    train_batches = train_data
    val_batches = valid_data

    for epoch in range(num_epochs):
        epoch_start = t.time()

        # Training
        model.train()
        for item_inputs, skill_inputs, item_ids, skill_ids, timestamp, label_inputs, labels in train_batches:
            rel = corr_data[(item_ids - 1).unsqueeze(1).repeat(1, item_ids.shape[-1], 1),
                            (item_inputs - 1).unsqueeze(-1).repeat(1, 1, item_inputs.shape[-1])]
            item_inputs = item_inputs.cuda()
            time = computeRePos(timestamp, timespan)
            skill_inputs = skill_inputs.cuda()
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            skill_ids = skill_ids.cuda()
            ren_mat = computeREN(skill_ids)

            preds, weights = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels.cuda(),
                                   torch.Tensor(rel).cuda(), ren_mat.cuda(), time.cuda())

            loss = compute_loss(preds, labels.cuda(), criterion)
            preds = torch.sigmoid(preds).detach().cpu()
            train_auc, train_acc = compute_auc(preds, labels)

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
            metrics.store({'loss/train': loss.item()})
            metrics.store({'auc/train': train_auc})

            # Logging
            if step % 1000 == 0:
                logger.log_scalars(metrics.average(), step)

        # Validation
        model.eval()
        for item_inputs, skill_inputs, item_ids, skill_ids, timestamp, label_inputs, labels in val_batches:
            rel = corr_data[(item_ids - 1).unsqueeze(1).repeat(1, item_ids.shape[-1], 1),
                            (item_inputs - 1).unsqueeze(-1).repeat(1, 1, item_inputs.shape[-1])]
            item_inputs = item_inputs.cuda()
            skill_inputs = skill_inputs.cuda()
            time = computeRePos(timestamp, timespan)
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            skill_ids = skill_ids.cuda()
            ren_mat = computeREN(skill_ids)

            with torch.no_grad():
                preds, weights = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels.cuda(),
                                       torch.Tensor(rel).cuda(), ren_mat.cuda(), time.cuda())

                preds = torch.sigmoid(preds).cpu()

            val_auc, val_acc = compute_auc(preds, labels)
            metrics.store({'auc/val': val_auc, 'acc/val': val_acc})

        # Save model
        cost_time = round((t.time() - epoch_start), 2)
        average_metrics = metrics.average()
        logger.log_scalars(average_metrics, step)

        print('epoch:', epoch, average_metrics, 'time:', cost_time)
        stop, best_epoch = saver.save(average_metrics['auc/val'], model, epoch)
        if stop:
            print("best epoch:", best_epoch)
            break


if __name__ == "__main__":
    """
    datasets：eanalyst, poj, junyi, ednet
    run model：python train_ykt.py --dataset poj --rel_pos
    """
    parser = argparse.ArgumentParser(description='Train YKT.')
    parser.add_argument('--dataset', type=str, default='poj')
    parser.add_argument('--logdir', type=str, default='runs/ykt')
    parser.add_argument('--savedir', type=str, default='save/ykt')
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--embed_size', type=int, default=60)
    parser.add_argument('--num_attn_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--rel_pos', action='store_true', default=False)
    parser.add_argument('--max_pos', type=int, default=10)
    parser.add_argument('--drop_prob', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=10)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--timespan', default=100000, type=int)
    parser.add_argument('--seed', type=int, default=2021, help='')
    parser.add_argument('--no_bert', action='store_true', default=False)

    args = parser.parse_args()
    print(args)

    set_random_seeds(args.seed)
    # divide train, valid, test set
    data_path = 'data/' + args.dataset + '.npz'
    pro_num, skill_num, train_data, valid_data, test_data = load_dataset(data_path, args.batch_size)
    print('train_batches: ', len(train_data), 'val_batches: ', len(valid_data), 'test_batches: ', len(test_data))
    corr_data = get_corr_data(args.dataset, pro_num)

    model = YKT(pro_num, skill_num, args.max_length, args.embed_size, args.num_attn_layers, args.num_heads,
                args.rel_pos, args.max_pos, args.drop_prob, args.no_bert, args.dataset).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)

    while True:
        param_str = (f'{args.dataset},'
                     f'batch_size={args.batch_size},'
                     f'max_length={args.max_length},'
                     f'rel_pos={args.rel_pos},'
                     f'max_pos={args.max_pos}')
        logger = Logger(os.path.join(args.logdir, param_str))
        saver = Saver(args.savedir, param_str)

        # train+valid
        train(train_data, valid_data, args.timespan, model, corr_data, optimizer, logger, saver,
              args.num_epochs, args.grad_clip)
        break

    logger.close()

    model = saver.load()
    test_batches = test_data

    # Predict on test set
    print('-----------------Testing-----------------')
    model.eval()
    test_preds = np.empty(0)
    correct = np.empty(0)
    for item_inputs, skill_inputs, item_ids, skill_ids, timestamp, label_inputs, labels in test_batches:
        rel = corr_data[(item_ids - 1).unsqueeze(1).repeat(1, item_ids.shape[-1], 1),
                        (item_inputs - 1).unsqueeze(-1).repeat(1, 1, item_inputs.shape[-1])]
        item_inputs = item_inputs.cuda()
        skill_inputs = skill_inputs.cuda()
        time = computeRePos(timestamp, args.timespan)
        label_inputs = label_inputs.cuda()
        item_ids = item_ids.cuda()
        skill_ids = skill_ids.cuda()
        ren_mat = computeREN(skill_ids)

        with torch.no_grad():
            preds, _ = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels.cuda(),
                             torch.Tensor(rel).cuda(), ren_mat.cuda(), time.cuda())
            preds = torch.sigmoid(preds[labels >= 0]).flatten().cpu().numpy()
            test_preds = np.concatenate([test_preds, preds])
        labels = labels[labels >= 0].float()
        correct = np.concatenate([correct, labels])

    print("auc_test = ", roc_auc_score(correct, test_preds))
    print("acc_test = ", accuracy_score(correct, test_preds.round()))
