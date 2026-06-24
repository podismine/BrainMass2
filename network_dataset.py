#coding:utf8
import os
from torch.utils import data
import numpy as np
import nibabel as nib
import random
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
from nilearn.connectome import ConnectivityMeasure

warnings.filterwarnings("ignore")
DATA_ROOT = "/work/yyang/brainmass2/preprocess/data_v2"
DF_PATH = '/work/yyang/brainmass2/preprocess/fmri_file_v2_each_file.csv'

def random_timeseries(timeser,sample_len):
    time_len = timeser.shape[1]
    st_thres = 1
    if time_len <= sample_len + st_thres:
        return timeser

    select_range = time_len - sample_len
    if select_range < 1:
        return timeser

    st = random.sample(list(np.arange(st_thres,select_range)),1)[0]
    return timeser[:,st:st+sample_len]

class Pretrain_data(data.Dataset):

    def __init__(self, root = DATA_ROOT,csv = DF_PATH, time_len=60):
        self.root = root
        self.time_len = time_len
        df = pd.read_csv(csv)
        self.names = list(df['npz_path'])

        print(f"Finding files: {len(self.names)}")
        self.correlation_measure = ConnectivityMeasure(kind='correlation')

    def map_sex(self, val):
        all_lists = ['Female',   'Male',      '1',      np.nan,      'M',      'F',   'male', 'female',      'f',      'm',  'Other',    '1.0',    '2.0',      'O']
        if val not in all_lists:
            return 1
        woman_lists = ['2.0','Female','F','f','female']
        man_lists = ['Male','M','m','1','male','1.0']
        if val in woman_lists:
            return 0
        else:
            return 1

    def __getitem__(self,index):
        name = self.names[index]
        img = np.load(os.path.join(self.root, name))

        ts = img['ts']#[:, :155]
        ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
        age = np.clip(float(img['age']), 0, 100) /100
        sex = self.map_sex(img['sex'])

        slices = [random_timeseries(img,sample_len=self.time_len).T, random_timeseries(img,sample_len=self.time_len).T]
        correlation_matrix = self.correlation_measure.fit_transform(slices)
        correlation_matrix[correlation_matrix!=correlation_matrix]=0
        bnw1 = np.arctanh(np.clip(correlation_matrix[0], -0.999999, 0.999999))
        bnw2 = np.arctanh(np.clip(correlation_matrix[1], -0.999999, 0.999999))
        return bnw1, bnw2, age, sex

    def __len__(self):
        return len(self.names)
        
class Task2Data(data.Dataset):

    def __init__(self, root= None, csv = None, mask_way='mask',mask_len=10, time_len=30,shuffle_seed=42,is_train = True, is_test = False):
        # self.template = 'sch'
        self.is_test = is_test
        self.is_train = is_train
        self.root = root

        self.mask_way = mask_way
        self.mask_len = mask_len
        self.time_len = time_len

        self.df = pd.read_csv(csv)

        self.names = list(self.df['file'])
        test_length = int(len(self.df) * 0.15)

        all_data = np.array(self.names)
        lbls = np.array(list([1 if f == 1 else 0 for f in self.df['dx'] ]))
        sites = np.array(self.df['site']) if 'site' in self.df.columns else lbls
        train_index = self.df[self.df['is_train']==1].index
        rest_index = self.df[self.df['is_train']==0].index

        data_train = all_data[train_index]
        labels_train = lbls[train_index]

        rest_data = all_data[rest_index]
        rest_site = sites[rest_index]
        rest_label = lbls[rest_index]


        split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_length, random_state=shuffle_seed)
        for valid_index, test_index in split2.split(rest_data, rest_site):
            data_test, labels_test = rest_data[test_index], rest_label[test_index]
            data_val, labels_val = rest_data[valid_index], rest_label[valid_index]

        if is_test is True:
            print("Testing data:")
            self.imgs, self.lbls = data_test, labels_test
        elif is_train is True:
            print("Training data:")
            self.imgs, self.lbls = data_train, labels_train
            # self.imgs, self.lbls = np.concatenate([data_train, data_val],0), np.concatenate([labels_train, labels_val],0),
        else:
            print("Val data:")
            self.imgs, self.lbls = data_val, labels_val
        print(self.imgs.shape)
        self.correlation_measure = ConnectivityMeasure(kind='correlation')


    def __getitem__(self,index):
        name = self.imgs[index]
        lbl = self.lbls[index]
        img = np.load(os.path.join(self.root, f"{name}.npy"))
        if self.is_train is True:
            slices = [random_timeseries(img,self.time_len).T]
            correlation_matrix = self.correlation_measure.fit_transform(slices).mean(0)
        elif self.is_test is False:
            slices = [img.T]
            correlation_matrix = self.correlation_measure.fit_transform(slices)[0]
        else:
            slices = [img.T]
            correlation_matrix = self.correlation_measure.fit_transform(slices).mean(0)
        onehot_lbl = np.zeros((2))
        onehot_lbl[lbl] = 1
        correlation_matrix[correlation_matrix!=correlation_matrix]=0
        return correlation_matrix,onehot_lbl, age, sex

    def __len__(self):
        return len(self.imgs)