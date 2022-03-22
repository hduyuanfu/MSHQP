from PIL import Image
from torch.utils.data import Dataset
import sys
sys.path.append('../../')
from MSHQP.config import car_train, car_test


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform, train=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.train = train

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        if self.train:
            img = self.loader(car_train + fn)
        else:
            img = self.loader(car_test + fn)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
