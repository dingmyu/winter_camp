import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils

from models import E1, E2, Decoder, Disc
from utils import load_model_for_eval 

from PIL import Image 
import functools

@functools.lru_cache(maxsize=1)
def get_transform(crop, resize):
    comp_transform = transforms.Compose([
#         transforms.CenterCrop(crop),
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return comp_transform

def default_loader(filepath):
    return Image.open(filepath).convert('RGB')

def load_image(filepath, crop, resize):
    image = default_loader(filepath)
    comp_transform = get_transform(crop, resize)
    image = comp_transform(image)
    return image

@functools.lru_cache(maxsize=1)
def get_eval_model(load, sep, resize):
    e1 = E1(sep, int((resize / 64)))
    e2 = E2(sep, int((resize / 64)))
    decoder = Decoder(int((resize / 64)))
    
    if torch.cuda.is_available():
        e1 = e1.cuda()
        e2 = e2.cuda()
        decoder = decoder.cuda()
    
    _iter = load_model_for_eval(load, e1, e2, decoder)
    
    e1 = e1.eval()
    e2 = e2.eval()
    decoder = decoder.eval()
    return e1, e2, decoder
    
def wear_glasses(load, sep, crop, resize, filepath, module_path):
    print('Start...')
    e1, e2, decoder = get_eval_model(load, sep, resize)
    module = load_image(module_path, crop, resize)
    image = load_image(filepath, crop, resize)
    
    with torch.no_grad():
        module = module.unsqueeze(0)
        image = image.unsqueeze(0)
        
        module = Variable(module)
        if torch.cuda.is_available():
            module = module.cuda()
            
        image = Variable(image)
        if torch.cuda.is_available():
            image = image.cuda()
        
        print("module shape : ", module.shape)
        print("image shape : ", image.shape)
        
        separate_A = e2(module)
        common_B = e1(image)
        BA_encoding = torch.cat([common_B, separate_A], dim=1)
        BA_decoding = decoder(BA_encoding)
    
    vutils.save_image(module, 'myimages/module.png', normalize=True)
    vutils.save_image(image, 'myimages/image.png', normalize=True)
    vutils.save_image(BA_decoding, 'myimages/wear_glasses.png', normalize=True)
    print('End!')

if __name__ == "__main__":
    """
    python wear_glasses.py --load /home/aailyk057pku/winter-camp-pek/cartoon/ContentDisentanglement/glasses_experiment_crop/checkpoint_10000 --filepath /home/aailyk057pku/winter-camp-pek/cartoon/cartoonset10k/cs10472220811414177401.png --module /home/aailyk057pku/winter-camp-pek/cartoon/cartoonset10k/cs1048392940779812096.png
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sep', type=int, default=25)
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--crop', type=int, default=178)
    parser.add_argument('--load', type=str, required=True)
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--module', type=str, required=True)
    args = parser.parse_args()
    
    wear_glasses(args.load, args.sep, args.crop, args.resize, args.filepath, args.module)
    
    
    