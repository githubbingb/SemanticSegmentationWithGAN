import torchvision.models as models


vgg16 = models.vgg16(pretrained=True)
pretrained_dict = vgg16.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items()}
print pretrained_dict