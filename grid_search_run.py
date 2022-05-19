# coding: UTF-8
from multiprocessing.connection import wait
import time, os
import numpy as np
from train_eval import train, test
import random
from bert import Model, Config
import argparse
from utils import build_dataset, build_iterator, get_time_dif, load_dataset, gettoken
from torch.utils.data import TensorDataset, DataLoader
import torch
import json
import pandas as pd

# get the pre_trained_model_dir
model_dir = os.path.join(os.getcwd(), 'bert_pretrain')
model_dir = ['bert_pretrain' + os.sep + i for i in os.listdir(model_dir) if i != '.DS_Store']
# exclude large model for now
model_dir = [i for i in model_dir if 'large' not in i]
# print('model_dir', model_dir)

# batch size
batch_size = [16, 32]
# learning rate
learning_rate = [1e-5, 5e-5, 1e-4, 1e-3]
# dropout
dropout = [0.1, 0.2, 0.3]
# epochs
epochs = [10, 20, 50]
# weight_decay
weight_decay = [0.1, 0.001, 0.0001]

hyper_size = len(model_dir) * len(batch_size) * len(learning_rate) * len(dropout) * len(epochs) * len(weight_decay)

arrays = ['Pre-trained', 'Batch', 'Learning R', 'Dropout', 'Weight Decay', 'Epochs', 'F1', 'Precision', 'Recall', 'Time Usage']
global df
df = pd.DataFrame(columns = arrays)
df['Model'] = [i for i in range(1, hyper_size + 1)]
df = df.set_index('Model')
df.to_csv('performance.csv', index = True)

def grid_search():
    global c
    c = 1
    for pre_train in model_dir:
        for b in batch_size:
            for l in learning_rate:
                for d in dropout:
                    for e in epochs:
                        for w in weight_decay:
                            parser = argparse.ArgumentParser(description='Salient triple classification')
                            parser.add_argument("--do_train", type=bool, default=True, help="Whether to run training.",)
                            parser.add_argument("--test_batch", default=200, type=int, help="Test every X updates steps.")

                            parser.add_argument("--data_dir", default="data", type=str, help="The task data directory.")
                            parser.add_argument("--model_dir", default=pre_train, type=str, help="The directory of pretrained models")
                            parser.add_argument("--output_dir", default='output/save_dict/', type=str, help="The path of result data and models to be saved.")
                            # models param
                            parser.add_argument("--max_length", default=256, type=int, help="the max length of sentence.")
                            parser.add_argument("--batch_size", default=b, type=int, help="Batch size for training.")
                            parser.add_argument("--learning_rate", default=l, type=float, help="The initial learning rate for Adam.")
                            parser.add_argument("--weight_decay", default=w, type=float, help="Weight decay if we apply some.")
                            parser.add_argument("--dropout", default=d, type=float, help="Drop out rate")
                            parser.add_argument("--epochs", default=e, type=int, help="Total number of training epochs to perform.")
                            parser.add_argument('--seed', type=int, default=1, help="random seed for initialization")
                            parser.add_argument('--hidden_size', type=int, default=768,  help="random seed for initialization")
                            args = parser.parse_args()

                            # @lw: record the hyper-parameters
                            df = pd.read_csv('performance.csv', index_col= 0)
                            df.loc[c, 'Pre-trained'] = pre_train.split(os.sep)[-1]
                            df.loc[c, 'Batch'] = b
                            df.loc[c, 'Learning R'] = l
                            df.loc[c, 'Dropout'] = d
                            df.loc[c, 'Epochs'] = e
                            df.loc[c, 'Weight Decay'] = w
                            # @lw: save the csv
                            df.to_csv('performance.csv', index = True)

                            global config
                            config = Config(args)

                            global model_name
                            model_name = '_'.join(list(map(str, [pre_train.split(os.sep)[-1], b, l, d, w, e])))

                            np.random.seed(args.seed)
                            torch.manual_seed(args.seed)
                            torch.cuda.manual_seed_all(args.seed)
                            torch.backends.cudnn.deterministic = True

                            train_entry()
                            c += 1


def train_entry():
    start_time = time.time()
    print("Loading data...")
    train_data_all = load_dataset(config.train_path, config)
    random.shuffle(train_data_all)
    offset = int(len(train_data_all) * 0.1)
    # @lw: build dev and train dataset
    dev_data = train_data_all[:offset]
    train_data = train_data_all[offset:]
    # @lw: load test dataset
    test_data = load_dataset(config.test_path, config)
    train_iter = DataLoader(
        train_data,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True)
    dev_iter = DataLoader(dev_data, shuffle=False, batch_size=config.batch_size,
                          num_workers=config.num_workers, drop_last=False)
    test_iter = DataLoader(test_data, shuffle=False, batch_size=config.batch_size,
                           num_workers=config.num_workers, drop_last=False)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # train
    model = Model(config).to(config.device)

    train(config, model, train_iter, dev_iter, test_iter, c)


# def test_entry():
#     test_data = load_dataset(config.test_path, config)
#     model = Model(config).to(config.device)

#     model.load_state_dict(torch.load(config.save_path + model_name + "model.ckpt"))
#     model.eval()
#     loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)
#     predicts = []
#     for i, batches in enumerate(loader):
#         sent, triple_id, _ = batches
#         input_ids, attention_mask, type_ids, position_ids = gettoken(config, sent)
#         input_ids, attention_mask, type_ids = \
#             input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device)
#         position_ids = position_ids.to(config.device)
#         pmi = model(input_ids, attention_mask, type_ids, position_ids)
#         bires = torch.where(pmi > 0.5, torch.tensor([1]).cuda(), torch.tensor([0]).cuda())
#         for b, t in zip(bires, triple_id):
#             predicts.append({"salience": b.item(), "triple_id": t})

#     with open(config.save_path + "xx_result.jsonl", "w") as f:
#         for t in predicts:
#             f.write(json.dumps(t, ensure_ascii=False)+"\n")


if __name__ == '__main__':
    grid_search()