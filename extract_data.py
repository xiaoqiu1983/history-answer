import json
import random
random.seed(42)

def read_data(filename):
    data = {}
    result = {}
    with open(filename, "r") as f:
        lines = f.readlines()
    for index, line in enumerate(lines):
        if index == 0:
            continue
        q, a, label = line.split("########")
        label = label.strip()
        if label == "问题":
            label = "内容"
        if label not in data:
            data[label] = []
            data[label].append((q,a))
        else:
            data[label].append((q,a))
    print(data.keys())
    for key in data.keys():
        print(len(data[key]))
        result[key] = random.sample(data[key], 80)
    return result

indexes = [i for i in range(80)]
sample_train_index = random.sample(indexes, 10)
for i in sample_train_index:
    indexes.remove(i)
sample_eval_index = indexes

_data = read_data("/home/gaokao.txt")
train_data = {}
eval_data = {}
for key in _data.keys():
    train_data[key] = []
    eval_data[key] = []
    for i in sample_train_index:
        train_data[key].append(_data[key][i])
    for i in sample_eval_index:
        eval_data[key].append(_data[key][i])

with open("eval_data.json", "w", encoding='utf-8') as f:
    f.write(json.dumps(eval_data))
with open("train_data.json", "w", encoding='utf-8') as f:
    f.write(json.dumps(train_data))
