import os
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DatasetLoader(Dataset):
    call_count = 0 

    def __init__(self, setname, args, train_augmentation=None):
        self.k_shot = args.k_shot
        self.query = args.query
        self.dalle_shot = args.dalle_shot

        DATASET_DIR = os.path.join(args.data_path, 'FC100')
        if setname == 'train':
            THE_PATH = osp.join(DATASET_DIR, 'train')
            label_list = os.listdir(THE_PATH)
        elif setname == 'test':
            THE_PATH = osp.join(DATASET_DIR, 'test')
            label_list = os.listdir(THE_PATH)
        elif setname == 'val':
            THE_PATH = osp.join(DATASET_DIR, 'val')
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Incorrect set name. Please check!')

        data = []
        label = []

        folders = sorted([osp.join(THE_PATH, label)
                         for label in label_list if os.path.isdir(osp.join(THE_PATH, label))])

        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        DATASET_DIR_DALLE = os.path.join(args.data_path, 'dalle_FC100')
        if setname == 'train':
            THE_PATH_DALLE = osp.join(DATASET_DIR_DALLE, 'train')
            label_list_dalle = os.listdir(THE_PATH_DALLE)
        elif setname == 'test':
            THE_PATH_DALLE = osp.join(DATASET_DIR_DALLE, 'test')
            label_list_dalle = os.listdir(THE_PATH_DALLE)
        elif setname == 'val':
            THE_PATH_DALLE = osp.join(DATASET_DIR_DALLE, 'val')
            label_list_dalle = os.listdir(THE_PATH_DALLE)

        data_dalle = []
        label_dalle = []

        folders_dalle = sorted([osp.join(THE_PATH_DALLE, label_dalle)
                               for label_dalle in label_list_dalle if os.path.isdir(osp.join(THE_PATH_DALLE, label_dalle))])

        for idx, this_folder in enumerate(folders_dalle):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data_dalle.append(osp.join(this_folder, image_path))
                label_dalle.append(idx)

        self.data_dalle = data_dalle
        self.label_dalle = label_dalle
        self.num_class_dalle = len(set(label_dalle))

        # Transformation
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        DatasetLoader.call_count += 1  
        if DatasetLoader.call_count % (self.k_shot + self.query + self.dalle_shot) <= self.dalle_shot:
            path, label = self.data_dalle[i], self.label_dalle[i]
            image = self.transform(Image.open(path).convert('RGB'))
        else:
            path, label = self.data[i], self.label[i]
            image = self.transform(Image.open(path).convert('RGB'))
        return image, label


if __name__ == '__main__':
    pass
