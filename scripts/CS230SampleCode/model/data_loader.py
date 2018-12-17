import random
import os
import numpy as np # THIS WAS ADDED BY US
import glob # THIS WAS ADDED BY US
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pydicom
import torch

SIZE = 64
# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    transforms.Resize(SIZE),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize(SIZE),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor


class SIGNSDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """	
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
#        self.filenames = os.listdir(data_dir)
#        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.dcm')]

#        filedir = '/data2/yeom/ky_mra'
#        pattern_match = '/**/**/**/**/*.dcm'
        #self.normals = glob.glob('/data2/yeom/ky_mra/Normal_MRA/**/**/**/**/*.dcm')
        #self.abnormals = glob.glob('/data2/yeom/ky_mra/MMD_MRA/**/**/**/**/*.dcm')
#        self.normals = glob.glob('/data2/yeom/ky_mra/Normal_MRA/**/**/**/*3D*/*.dcm')
#        self.abnormals = glob.glob('/data2/yeom/ky_mra/MMD_MRA/**/**/**/*3D**/*.dcm')
#        self.filenames = self.normals + self.abnormals 

#        self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
#        self.data = np.loadtxt('/home/ky_mra/CS230Fall2018MRA/scripts/all_data_matrix.txt', dtype=float)
        self.data = np.loadtxt('/home/ky_mra/CS230Fall2018MRA/scripts/all_data_matrix_256_MRAonly.txt', dtype = np.uint8)
        self.data = np.reshape(self.data, (self.data.shape[0], 256, 256, 3))
#        print(self.data.shape)
#        print(self.data[1].shape)
#        self.data = np.mean(self.data, axis=3) 
#        print(self.data.shape)
#        print(self.data[1].shape)
#        self.labels = np.concatenate((np.zeros((len(self.normals),)), np.ones((len(self.abnormals), ))))
        #self.labels = np.loadtxt('/home/ky_mra/CS230Fall2018MRA/scripts/all_labels.txt', dtype = float)
        self.labels = np.loadtxt('/home/ky_mra/CS230Fall2018MRA/scripts/all_labels_MRAonly.txt', dtype=np.uint8)
        self.transform = transform

#        self.count = 0

    def __len__(self):
        # return size of dataset
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
   
        # image = Image.open(self.filenames[idx])  # PIL image
#        try: 
#            print(self.filenames[idx])
#            image = pydicom.dcmread(self.filenames[idx]).pixel_array  #dcm image
#            image = Image.fromarray(image) #TODO: check if image from dicom is normalized in the right way for convertion to PIL (e.g. 0-1 or 0-255)
#            image = self.transform(image)
#            return image, self.labels[idx]
#        except Exception:
#            self.count += 1
#            print(self.count, self.filenames[idx])
            #print(">>>skipped", self.filenames[idx+1])
            #image = pydicom.dcmread(self.filenames[idx+1]).pixel_array  #dcm image
        image = Image.fromarray(self.data[idx]) #TODO: check if image from dicom is normalized in the right way for convertion to PIL (e.g. 0-1 or 0-255)
        image = self.transform(image)
#        print(image.shape)

        labels = self.labels[idx]
#        labels = torch.from_numpy(np.array(labels))
        labels = np.array(labels)

        #return image, self.labels[idx]
        return image, labels
            #return image, self.labels[idx]

#            pass

def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_signs".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(SIGNSDataset(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(SIGNSDataset(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
