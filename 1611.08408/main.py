from DataFolder import MyDataFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


dataFolder = MyDataFolder(data_root='/media/Disk/work', txt='/media/Disk/work/train.txt',
                          transform=transforms.ToTensor())
dataloader = DataLoader(dataset=dataFolder, batch_size=1, shuffle=False, num_workers=2)



def onehot_encoder(ground_truth):
    for index, c in enumerate(range(0, 2)):
        mask = (ground_truth == c)
        mask.view(-1, 1, mask.shape[0], mask.shape[1])

        if index == 0:
            onehot = mask
        else:
            onehot = torch.cat(onehot, mask)

        return onehot.float()

for i, data in enumerate(dataloader):
    img, label = data
    print len(dataloader)