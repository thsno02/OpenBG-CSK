import random
import jsonlines, os

random.seed(2022)

cwd = os.getcwd()
data_path = os.path.join(cwd, 'data')
file_path = os.path.join(data_path, 'train_triple.jsonl')

train_data = []

with jsonlines.open(file_path, 'r') as f:
    for item in f:
        train_data.append(item)

random.shuffle(train_data)

dev_size = int(len(train_data) * 0.4)

dev_data = train_data[:dev_size]
train_data = train_data[dev_size:]

with jsonlines.open(file_path, 'w') as writer:
    writer.write_all(train_data)

file_path = os.path.join(data_path, 'dev_triple.jsonl')
with jsonlines.open(file_path, 'w') as writer:
    writer.write_all(dev_data)