import os
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class tieredImageNet(Dataset):
    call_count = 0  

    def __init__(self, setname, args=None, train_augmentation=None):
        self.k_shot = args.k_shot
        self.query = args.query
        self.dalle_shot = args.dalle_shot

        TRAIN_PATH = osp.join(args.data_path, 'tiered_imagenet/train')
        VAL_PATH = osp.join(args.data_path, 'tiered_imagenet/val')
        TEST_PATH = osp.join(args.data_path, 'tiered_imagenet/test')
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        elif setname == 'test':
            THE_PATH = TEST_PATH
        elif setname == 'val':
            THE_PATH = VAL_PATH
        else:
            raise ValueError('Incorrect set name. Please check!')
        data = []
        label = []
        folders = sorted([osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
                          os.path.isdir(osp.join(THE_PATH, label))])

        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        TRAIN_PATH_DALLE = osp.join(
            args.data_path, 'dalle_tiered_imagenet/train')
        VAL_PATH_DALLE = osp.join(args.data_path, 'dalle_tiered_imagenet/val')
        TEST_PATH_DALLE = osp.join(
            args.data_path, 'dalle_tiered_imagenet/test')
        if setname == 'train':
            THE_PATH_DALLE = TRAIN_PATH_DALLE
        elif setname == 'test':
            THE_PATH_DALLE = TEST_PATH_DALLE
        elif setname == 'val':
            THE_PATH_DALLE = VAL_PATH_DALLE

        data_dalle = []
        label_dalle = []
        folders_dalle = sorted([osp.join(THE_PATH_DALLE, label_dalle) for label_dalle in os.listdir(THE_PATH_DALLE) if
                                os.path.isdir(osp.join(THE_PATH_DALLE, label_dalle))])

        for idx in range(len(folders_dalle)):
            this_folder = folders_dalle[idx]
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data_dalle.append(osp.join(this_folder, image_path))
                label_dalle.append(idx)

        self.data_dalle = data_dalle
        self.label_dalle = label_dalle
        self.num_class_dalle = len(set(label_dalle))

        # Transformation
        if (setname == 'val' or setname == 'test' or setname == 'train') and train_augmentation is None:
            image_size = args.image_size
            if image_size == 224:
                img_resize = 256
            elif image_size == 84:
                img_resize = 92
            else:
                ValueError('Image size not supported at the moment.')
            self.transform = transforms.Compose([
                transforms.Resize([img_resize, img_resize]),
                transforms.CenterCrop(image_size),
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
        tieredImageNet.call_count += 1  
        if tieredImageNet.call_count % (self.k_shot + self.query + self.dalle_shot) <= self.dalle_shot:
            path, label = self.data_dalle[i], self.label_dalle[i]
            image = self.transform(Image.open(path).convert('RGB'))
        else:
            path, label = self.data[i], self.label[i]
            image = self.transform(Image.open(path).convert('RGB'))
        return image, label


if __name__ == '__main__':
    pass
