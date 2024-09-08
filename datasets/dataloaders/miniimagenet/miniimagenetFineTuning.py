import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os


class MiniImageNet(Dataset):
    call_count = 0  

    def __init__(self, setname, args, train_augmentation=None):
        self.k_shot = args.k_shot
        self.query = args.query
        self.dalle_shot = args.dalle_shot

        IMAGE_PATH = os.path.join(args.data_path, 'mini_imagenet/images')
        SPLIT_PATH = os.path.join(args.data_path, 'mini_imagenet/split')

        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data  # data path of all data
        self.label = label  # label of all data
        self.num_class = len(set(label))

        IMAGE_PATH_DALLE = os.path.join(
            args.data_path, 'dalle_mini_imagenet/images')
        SPLIT_PATH_DALLE = os.path.join(
            args.data_path, 'dalle_mini_imagenet/split')

        csv_path_dalle = osp.join(SPLIT_PATH_DALLE, setname + '.csv')
        lines_dalle = [x.strip()
                       for x in open(csv_path_dalle, 'r').readlines()][1:]

        data_dalle = []
        label_dalle = []
        lb_dalle = -1

        self.wnids_dalle = []

        for l_dalle in lines_dalle:
            name_dalle, wnid_dalle = l_dalle.split(',')
            path_dalle = osp.join(IMAGE_PATH_DALLE, name_dalle)
            if wnid_dalle not in self.wnids_dalle:
                self.wnids_dalle.append(wnid_dalle)
                lb_dalle += 1
            data_dalle.append(path_dalle)
            label_dalle.append(lb_dalle)

        self.data_dalle = data_dalle  # data path of all data
        self.label_dalle = label_dalle  # label of all data
        self.num_class_dalle = len(set(label_dalle))

        if (setname == 'val' or setname == 'test' or setname == 'train') and train_augmentation is None:
            self.transform = transforms.Compose([
                # transforms.Resize([92, 92]),
                transforms.Resize([256, 256]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                #                      np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))  # ImageNet standard
            ])
        elif setname == 'train' and train_augmentation is not None:
            self.transform = train_augmentation
        else:
            ValueError("Set name or train augmentation corrupt. Please check!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        MiniImageNet.call_count += 1 
        if MiniImageNet.call_count % (self.k_shot + self.query + self.dalle_shot) <= self.dalle_shot:
            path, label = self.data_dalle[i], self.label_dalle[i]
            image = self.transform(Image.open(path).convert('RGB'))
        else:
            path, label = self.data[i], self.label[i]
            image = self.transform(Image.open(path).convert('RGB'))
        return image, label


if __name__ == '__main__':
    pass
