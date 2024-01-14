import argparse
# from path import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
from tqdm import tqdm

import torchvision.transforms as transforms
import flow_transforms
from imageio import imread, imwrite
import numpy as np
from util import flow2rgb

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


# parser = argparse.ArgumentParser(description='PyTorch FlowNet inference on a folder of img pairs',
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('data', metavar='DIR',
#                     help='path to images folder, image names must match \'[name]0.[ext]\' and \'[name]1.[ext]\'')
# parser.add_argument('pretrained', metavar='PTH', help='path to pre-trained model')
# parser.add_argument('--output', '-o', metavar='DIR', default=None,
#                     help='path to output folder. If not set, will be created in data folder')
# parser.add_argument('--output-value', '-v', choices=['raw', 'vis', 'both'], default='both',
#                     help='which value to output, between raw input (as a npy file) and color vizualisation (as an image file).'
#                     ' If not set, will output both')
# parser.add_argument('--div-flow', default=20, type=float,
#                     help='value by which flow will be divided. overwritten if stored in pretrained file')
# parser.add_argument("--img-exts", metavar='EXT', default=['png', 'jpg', 'bmp', 'ppm'], nargs='*', type=str,
#                     help="images extensions to glob")
# parser.add_argument('--max_flow', default=None, type=float,
#                     help='max flow value. Flow map color is saturated above this value. If not set, will use flow map\'s max value')
# parser.add_argument('--upsampling', '-u', choices=['nearest', 'bilinear'], default=None, help='if not set, will output FlowNet raw input,'
#                     'which is 4 times downsampled. If set, will output full resolution flow map, with selected upsampling')
# parser.add_argument('--bidirectional', action='store_true', help='if set, will output invert flow (from 1 to 0) along with regular flow')
#
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    # global args, save_path
    # args = parser.parse_args()
    #
    # if args.output_value == 'both':
    #     output_string = "raw output and RGB visualization"
    # elif args.output_value == 'raw':
    #     output_string = "raw output"
    # elif args.output_value == 'vis':
    #     output_string = "RGB visualization"
    # print("=> will save " + output_string)
    # data_dir = Path(args.data)
    # print("=> fetching img pairs in '{}'".format(args.data))
    # if args.output is None:
    #     save_path = data_dir/'flow'
    # else:
    #     save_path = Path(args.output)
    # print('=> will save everything to {}'.format(save_path))
    # save_path.makedirs_p()
    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])


    img_pairs = [("F:\project_old\FlowNetPytorch\images\original.jpg","F:\project_old\FlowNetPytorch\images\modified.jpg")]


    # for ext in args.img_exts:
        # test_files = data_dir.files('*1.{}'.format(ext))
        # for file in test_files:
        #     img_pair = file.parent / (file.stem[:-1] + '2.{}'.format(ext))
        #     if img_pair.isfile():
        #         img_pairs.append([file, img_pair])

    # print('{} samples found'.format(len(img_pairs)))
    # create model
    network_data = torch.load("F:/project_old/FlowNetPytorch/pytorch/flownets_bn_EPE2.459.pth.tar")
    # print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](network_data).to(device)
    model.eval()
    cudnn.benchmark = True

    # if 'div_flow' in network_data.keys():
    #     args.div_flow = network_data['div_flow']

    for (img1_file, img2_file) in tqdm(img_pairs):

        img1 = input_transform(imread(img1_file))
        img2 = input_transform(imread(img2_file))
        input_var = torch.cat([img1, img2]).unsqueeze(0)

        # if args.bidirectional:
        #     # feed inverted pair along with normal pair
        #     inverted_input_var = torch.cat([img2, img1]).unsqueeze(0)
        #     input_var = torch.cat([input_var, inverted_input_var])

        input_var = input_var.to(device)
        # compute output
        output = model(input_var)

        print(output.shape)
        # if args.upsampling is not None:
        #     output = F.interpolate(output, size=img1.size()[-2:], mode=args.upsampling, align_corners=False)
        for suffix, flow_output in zip(['flow', 'inv_flow'], output):

            # flow = np.transpose(flow_output, (1, 2, 0))
            # flow = np.transpose(flow_output.data.cpu(), (1, 2, 0))
            # zero_flow=np.zeros(shape=flow.shape)
            # import cv2
            # cv2.imshow("",flow)
            # cv2.waitKey(10000)

            # h, w, _ = flow.shape
            # print(w, h)

            # Undoing the warps
            # modified = face.resize((w, h), Image.BICUBIC)

            # modified_np = np.asarray(modified)

            # filename = save_path/'{}{}'.format(img1_file.stem[:-1], suffix)
            if "vis" in['vis', 'both']:
                rgb_flow = flow2rgb(flow_output, max_value=None)
                to_save = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)
                imwrite("output_test" + '.png', to_save)


if __name__ == '__main__':

    main()



#
# Given the original image Xorig
# and manipulated image Xmod, we compute the flow from
# original to manipulated and from manipulated to original
# using PWC-Net [33], which we denote Uom and Umo, respectively.
# To compute the flow consistency mask, we transform
# Umo from the manipulated image space into the original
# image space, which is U
# 0
# mo = T
#
# Umo; Uom
# . We consider the flow to be consistent at a pixel if the magnitude of
# U
# 0
# mo + Uom is less than a threshold. After this test, pixels
# corresponding to occlusions and ambiguities (e.g., in lowtexture regions) will be marked as inconsistent, and therefore do not contribute to the loss.
# We take relative error of the flow consistency as the criterion. For a pixel p,
# Minconsistent(p) = 1
# ||U
# 0
# mo(p) + Uom(p)||2
# ||Uom(p)||2 + 
# > τ
# . (4)
# We take  = 0.1 and τ = 0.85, then apply a Gaussian
# blur with σ = 7, denoted by G, and take the complement to
# get the flow consistency mask M:
# M = 1 − G(Minconsistent) (5)