import os
import json
import argparse
from utils.helper import set_seed



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create Experimental Config")
    parser.add_argument("--data_root", default="data", type=str)
    parser.add_argument("--dataset", default="mvtec", type=str)
    parser.add_argument("--category", default='hazelnut', type=str)
    parser.add_argument("--normal_k_shot", default=2, type=int)
    parser.add_argument("--anomaly_k_shot", default=0, type=int)
    parser.add_argument("--experiment_root", default="experiment", type=str)

    args = parser.parse_args()

    experiment_root = os.path.join(args.experiment_root, args.dataset, args.category)
    data_root = os.path.join(args.data_root, args.dataset, 'samples')

    train_config = os.path.join(data_root, "train_{}.jsonl".format(args.category))
    with open(train_config, 'r+') as f:
        samples = [json.loads(line) for line in f.readlines()]

    anomaly_types = {}
    for sample in samples:
        if sample['label_name'] not in anomaly_types:
            anomaly_types[sample['label_name']] = []
        anomaly_types[sample['label_name']].append(sample)

    assert 'normal' in anomaly_types

    choice_samples = []

    if args.normal_k_shot != 0:
        choice_samples.extend(anomaly_types['normal'][:args.normal_k_shot])

    if args.anomaly_k_shot != 0:
        for anomaly_type in anomaly_types:
            if anomaly_type != 'normal':
                choice_samples.extend(anomaly_types[anomaly_type][:args.anomaly_k_shot])

    if args.anomaly_k_shot == 0:
        config_name = "zero_{}_n{}.jsonl".format(args.category, args.normal_k_shot)
    else:
        config_name = "few_{}_n{}a{}.jsonl".format(args.category, args.normal_k_shot, args.anomaly_k_shot)

    save_root = os.path.join(experiment_root, 'samples')
    os.makedirs(save_root, exist_ok=True)

    with open(os.path.join(save_root, config_name), 'w+') as f:
        choice_samples = [json.dumps(sample)+"\n" for sample in choice_samples]
        f.writelines(choice_samples)
