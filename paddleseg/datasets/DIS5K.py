# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import json

import paddle
from skimage import io
import os

from tqdm import tqdm

from paddleseg.cvlibs import manager
from paddleseg.datasets import Dataset
from paddleseg.transforms import Compose,gt_preprocess,im_preprocess

@manager.DATASETS.add_component
class DIS5K(Dataset):
    NUM_CLASSES = 1

    def __init__(self, transforms,
                 dataset_root=None,
                 cache_path='./DIS5K-Cache',
                 cache_file_name='dataset.json',
                 cache_size=[1024,1024],
                 mode='train',
                 edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.edge = edge
        self.cache_path = cache_path
        self.cache_size = cache_size
        self.cache_file_name=cache_file_name
        if mode not in ['train', 'test', 'val']:
            raise ValueError(
                "`mode` should be one of ('train', 'trainval', 'trainaug', 'val') in PascalVOC dataset, but got {}."
                    .format(mode))
        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        img_dir = os.path.join(self.dataset_root, 'im')
        label_dir = os.path.join(self.dataset_root, 'gt')

        if self.dataset_root is None or not os.path.isdir(
                self.dataset_root) or not os.path.isdir(
            img_dir) or not os.path.isdir(label_dir):
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )

        label_files = sorted(
            glob.glob(label_dir + os.sep + '*' + '.png'))
        img_files = sorted(
            glob.glob(img_dir + os.sep + '*' + '.jpg'))

        self.file_list = [
            [img_path, label_path]
            for img_path, label_path in zip(img_files, label_files)
        ]
        ims_pt_list = []
        gts_pt_list = []
        # 第二级地址
        cache_folder_1 = os.path.join(self.cache_path, dataset_root.split("/")[-1])
        if not os.path.exists(cache_folder_1):
            os.makedirs(cache_folder_1)
        cache_folder = os.path.join(cache_folder_1,dataset_root.split("/")[-1]+ "_" + "x".join([str(x) for x in self.cache_size]))
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
            for i, path in tqdm(enumerate(self.file_list), total=len(self.file_list)):
                im_path = path[0]
                im = io.imread(im_path)
                im = im_preprocess(im, self.cache_size)
                im_id = im_path.split("/")[-1].split(".")[0]
                im_cache_file = os.path.join(cache_folder, im_id + "_im.pdparams")
                paddle.save(im, im_cache_file)
                ims_pt_list.append(im_cache_file)

                gt_path = path[1]
                gt = io.imread(gt_path)
                gt = gt_preprocess(gt, self.cache_size)
                gt_id = gt_path.split("/")[-1].split(".")[0]
                gt_cache_file = os.path.join(cache_folder, gt_id + "_gt.pdparams")
                paddle.save(gt, gt_cache_file)
                gts_pt_list.append(gt_cache_file)

            self.file_list = [
                [img_path, label_path]
                for img_path, label_path in zip(ims_pt_list, gts_pt_list)
            ]

            json_file = open(os.path.join(cache_folder_1, self.cache_file_name), "w")
            json.dump(self.file_list, json_file)
            json_file.close()
        else:
            
            json_file = open(os.path.join(cache_folder_1, self.cache_file_name), "r")
            dataset = json.load(json_file)
            json_file.close()
            self.file_list = dataset
