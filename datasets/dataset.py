import os
import random
import PIL
import torch
import json
from PIL import Image
import numpy as np
from torch.utils.data import BatchSampler


class AnomalyClassificationDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_root,
        experiment_root,
        dataset,
        category,
        split,
        sample_file,
        use_gen,
        image_size,
        transforms,
        class_list = None,
        combined_name = 'combined',
        **kwargs,
    ):

        super().__init__()
        assert split in ['train', 'test']
        self.data_root = data_root
        self.experiment_root = experiment_root
        self.dataset = dataset
        self.category = category
        self.split = split
        self.sample_file = sample_file
        self.image_size = image_size
        self.transforms = transforms
        self.use_gen = use_gen
        self.class_list = class_list
        self.combined_name = combined_name
        self.get_image_data()

    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]

        is_synthesize = info['synthesize'] if 'synthesize' in info else False
        if is_synthesize:
            image_path = info['filename']
        else:
            image_path = os.path.join(self.data_root, self.dataset, info['filename'])

        image = self.read_image(image_path)

        if 'maskname' not in info:
            dt_mask = torch.zeros(image.shape[1:], dtype=torch.float)
            cls_mask = torch.zeros(image.shape[1:], dtype=torch.float)
        else:
            if is_synthesize:
                mask_path = info['maskname']
                mask = PIL.Image.open(mask_path).convert("L")
                mask = np.array(mask.resize((self.image_size, self.image_size), resample=PIL.Image.Resampling.NEAREST))
                dt_mask = mask.copy()
                dt_mask[dt_mask != 0] = 1
                cls_mask = mask.copy()
                cls_mask[cls_mask != 0] = info['label']
            else:
                mask_path = os.path.join(self.data_root, self.dataset, info['maskname'])
                mask = PIL.Image.open(mask_path).convert("L")
                mask = np.array(mask.resize((self.image_size, self.image_size), resample=PIL.Image.Resampling.NEAREST))
                dt_mask = mask.copy()
                dt_mask[dt_mask != 0] = 1
                cls_mask = mask.copy()
                assert np.any(cls_mask == info['label']) or info['label_name'] == self.combined_name

            dt_mask = torch.from_numpy(dt_mask).float()
            cls_mask = torch.from_numpy(cls_mask).float()

        cls_mask = cls_mask - 1 # normal is -1, anomaly in [0, Y-1]

        return {
            "image": image,
            "dt_mask": dt_mask,
            "cls_mask": cls_mask,
            'paths': image_path,
            "anomaly_type": info['label'],
            "is_synthesize": is_synthesize,
            "is_combined": info['label_name'] == self.combined_name
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def read_image(self, path):
        image = PIL.Image.open(path).convert("RGB")
        image = self.transforms(image)
        return image

    def get_image_data(self):
        self.data_to_iterate = []

        if self.split == 'train':
            sample_root = os.path.join(self.experiment_root, self.dataset, self.category, 'samples')
            self.real_data_to_iterate = []
            with open(os.path.join(sample_root, self.sample_file), "r") as f_r:
                for line in f_r:
                    sample = json.loads(line)
                    if sample['label_name'] != self.combined_name:
                        sample.update({'synthesize': False})
                        self.real_data_to_iterate.append(sample)
            self.data_to_iterate.extend(self.real_data_to_iterate)

            if self.use_gen:
                self.synthesize_data_to_iterate = []
                gen_files = [os.path.join(sample_root, file) for file in os.listdir(sample_root)
                             if file.find(self.sample_file.split('.')[0]) != -1 and file.find('gen') != -1]
                for file in gen_files:
                    if file.find(self.combined_name) == -1:
                        samples = []
                        with open(file, "r") as f_r:
                            for line in f_r:
                                sample = json.loads(line)
                                sample.update({'synthesize': True})
                                samples.append(sample)
                        print("find gened sample file {}, load {} samples".format(file, len(samples)))
                        self.synthesize_data_to_iterate.extend(samples)
                self.data_to_iterate.extend(self.synthesize_data_to_iterate)

        else:
            with open(os.path.join(self.data_root, self.dataset, 'samples', "{}_{}.jsonl".format(self.split, self.category)), "r") as f_r:
                for line in f_r:
                    self.data_to_iterate.append(json.loads(line))

        if self.class_list is None:
            self.class_list = {}
            for sample in self.data_to_iterate:
                if sample['label_name'] not in self.class_list:
                    self.class_list[sample['label_name']] = sample['label']
            self.class_list = [[key, self.class_list[key]] for key in self.class_list]
            self.class_list.sort(key=lambda item: item[-1])

        for sample in self.data_to_iterate:
            if 'label' not in sample:
                for class_name, class_id in self.class_list:
                    if class_name == sample['label_name']:
                        sample.update({'label': class_id})
                        break

        choice_samples = {label_name[0]: [] for label_name in self.class_list}

        if self.combined_name in [c[0] for c in self.class_list]:
            assert self.class_list[-1][0] == self.combined_name
            self.class_list = self.class_list[:-1]
            print('find combined, does not participate in the classification')

        for sample in self.data_to_iterate:
            choice_samples[sample['label_name']].append(sample)

        for label_name in choice_samples:
            print("{} {} has sample number is {}".format(self.split.upper(), label_name, len(choice_samples[label_name])))



class FixedInBatchSampler(BatchSampler):

    def __init__(self, dataset, batch_size, num_steps_per_epoch, is_zero_shot):

        assert dataset.split == 'train'
        self.dataset = dataset
        self.batch_size = batch_size
        self.is_zero_shot = is_zero_shot
        self.num_steps_per_epoch = num_steps_per_epoch
        self.real_samples = getattr(self.dataset, 'real_data_to_iterate', None)
        self.synthesize_samples = getattr(self.dataset, 'synthesize_data_to_iterate', None)
        self.real_sample_number = min(self.batch_size, len(self.real_samples))

    def __iter__(self):
        for _ in range(self.num_steps_per_epoch):
            if self.is_zero_shot:
                data_to_iterate = getattr(self.dataset, 'data_to_iterate')
                batch = random.sample(list(range(len(data_to_iterate))), min(self.batch_size, len(data_to_iterate)))
            else:
                if self.synthesize_samples is None:
                    batch = random.sample(list(range(len(self.real_samples))), self.real_sample_number)
                else:
                    batch = random.sample(list(range(len(self.real_samples))), self.real_sample_number)
                    if len(batch) < self.batch_size:
                        batch += random.sample(list(range(len(self.real_samples), len(self.real_samples) + len(self.synthesize_samples))),
                                               min(self.batch_size - len(batch), len(self.synthesize_samples)))
            yield batch

    def __len__(self):
        return self.num_steps_per_epoch