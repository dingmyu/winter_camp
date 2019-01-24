import argparse
import os
import torch
from models import E1, E2, Decoder
from utils import load_model_for_eval, get_test_imgs
import torchvision.utils as vutils

def trans(args, rc_e1, rc_e2, rc_decoder, c_e1, c_e2, c_decoder):
    test_domA, test_domB = get_test_imgs(args)
    
    exps = []

    # --------------- real images --------------- #   
    for i in range(args.num_display):
        with torch.no_grad():
            exps.append(test_domA[i].unsqueeze(0))
    
    # --------------- real2cartoon images --------------- #
    intput_cartoons = []
    for i in range(args.num_display):
            separate_A = torch.full((1, args.sep * (args.resize // 64) * (args.resize // 64)), 0).cuda()
            common_A = rc_e1(test_domA[i].unsqueeze(0))
            A_encoding = torch.cat([common_A, separate_A], dim=1)
            A_decoding = rc_decoder(A_encoding)
            exps.append(A_decoding)
            intput_cartoons.append(A_decoding)
            
    # --------------- cartoon2cartoon images --------------- #
    output_cartoons = []
    for i in range(args.num_display):
            separate_A = c_e2(test_domB[i].unsqueeze(0))
            common_B = c_e1(intput_cartoons[i])
            BA_encoding = torch.cat([common_B, separate_A], dim=1)
            BA_decoding = c_decoder(BA_encoding)
            exps.append(BA_decoding)
            output_cartoons.append(BA_decoding)
            
    # --------------- cartoon2real images --------------- #
    for i in range(args.num_display):
            separate_A = rc_e2(test_domA[i].unsqueeze(0))
            common_B = rc_e1(output_cartoons[i])
            BA_encoding = torch.cat([common_B, separate_A], dim=1)
            BA_decoding = rc_decoder(BA_encoding)
            exps.append(BA_decoding)
            
    # ------------- reference cartoon images ------------- #   
    for i in range(args.num_display):
        with torch.no_grad():
            exps.append(test_domB[i].unsqueeze(0))
    
    with torch.no_grad():
        exps = torch.cat(exps, 0)
        
    vutils.save_image(exps,
                      '%s/experiments.png' % (args.out),
                      normalize=True, nrow=args.num_display)
            
    
def test(args):
    # ---------- load model_real_cartoon ---------- #
    
    rc_e1 = E1(args.sep, int((args.resize / 64)))
    rc_e2 = E2(args.sep, int((args.resize / 64)))
    rc_decoder = Decoder(int((args.resize / 64)))

    if torch.cuda.is_available():
        rc_e1 = rc_e1.cuda()
        rc_e2 = rc_e2.cuda()
        rc_decoder = rc_decoder.cuda()

    if args.load_rc != '':
        save_file = os.path.join(args.load_rc)
        load_model_for_eval(save_file, rc_e1, rc_e2, rc_decoder)

    rc_e1 = rc_e1.eval()
    rc_e2 = rc_e2.eval()
    rc_decoder = rc_decoder.eval()
    
    # ---------- load model_cartoon ---------- #
    
    c_e1 = E1(args.sep, int((args.resize / 64)))
    c_e2 = E2(args.sep, int((args.resize / 64)))
    c_decoder = Decoder(int((args.resize / 64)))

    if torch.cuda.is_available():
        c_e1 = c_e1.cuda()
        c_e2 = c_e2.cuda()
        c_decoder = c_decoder.cuda()

    if args.load_c != '':
        save_file = os.path.join(args.load_c)
        load_model_for_eval(save_file, c_e1, c_e2, c_decoder)

    c_e1 = c_e1.eval()
    c_e2 = c_e2.eval()
    c_decoder = c_decoder.eval()
    
    # -------------- running -------------- #
    
    if not os.path.exists(args.out) and args.out != "":
        os.mkdir(args.out)

    trans(args, rc_e1, rc_e2, rc_decoder, c_e1, c_e2, c_decoder)

if __name__=='__main__':
    """shell
    python joint.py --load_rc /home/aailyk057pku/winter-camp-pek/model/checkpoint --load_c /home/aailyk057pku/winter-camp-pek/model/checkpoint_40000
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data_joint')
    parser.add_argument('--load_rc', default='')
    parser.add_argument('--load_c', default='')
    parser.add_argument('--out', default='joint')
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--cropA', type=int, default=178)
    parser.add_argument('--cropB', type=int, default=378)
    parser.add_argument('--sep', type=int, default=25)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--num_display', type=int, default=20)

    args = parser.parse_args()

    test(args)
    