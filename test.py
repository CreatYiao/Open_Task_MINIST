import torch
import torchvision.transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from main import *

img_path = "data10.png"
img = Image.open(img_path).convert("1")  # 转成单通道
# img.show()
# print(img)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((28, 28)), torchvision.transforms.ToTensor()])

img = transform(img)

# print(img.size())

model = torch.load("mnist_self_nor.pth")

img = torch.reshape(img, (1, 1, 28, 28))

# writer = SummaryWriter("logs")
# writer.add_images("test1", img, 1)

model.eval()

img = img.cuda()  # 放到GPU中训练

with torch.no_grad():
    output = model(img)

# writer.close()

# print(output)
print(output.argmax(1))