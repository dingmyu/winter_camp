from PIL import Image
import torchvision.transforms as transforms

real=Image.open('celeba/004806.jpg')
cart=Image.open('cartoonset10k/cs10472220811414177401.png')

print(real.size,cart.size)

comp_transformA = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(128)
    ])
    
comp_transformB = transforms.Compose([
        transforms.CenterCrop(378),
        transforms.Resize(128)
    ])
    
A = comp_transformA(real)
B = comp_transformB(cart)

A.save('A.jpg')
B.save('B.png')