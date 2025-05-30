from torch.utils.data import Dataset
from torchvision import transforms as T
import os
from PIL import Image

TRANSFORM = T.Compose([T.Resize((64, 64)), T.ToTensor(), T.Normalize(mean=[0.4714, 0.4411, 0.3918], std=[0.2753, 0.2679, 0.2791])])


class TinyImageNetDataset(Dataset):
    def __init__(
        self,
        root_dir_path="data/tiny-imagenet-200/{split}/images",
        split="test",
        transform=None,
    ):
        assert split in ["val", "test"]
        self.dir_path = root_dir_path.format(split=split)
        self.files = os.listdir(self.dir_path)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.dir_path, self.files[index])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
