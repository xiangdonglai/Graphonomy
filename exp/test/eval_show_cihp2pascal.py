import socket
import timeit
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
import glob
from collections import OrderedDict
sys.path.append('../../')
# PyTorch includes
import torch
import pdb
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import cv2

# Tensorboard include
# from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import pascal
from utils import util
from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr

#
import argparse
import copy
import torch.nn.functional as F
from test_from_disk import eval_


gpu_id = 1

label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0), (0,128,0), (128,128,0), (0,0,128), (128,0,128), (0,128,128)]

def img_transform(img, transform=None):
    sample = {'image': img, 'label': 0}
    sample = transform(sample)
    return sample


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

# def flip_cihp(tail_list):
#     '''

#     :param tail_list: tail_list size is 1 x n_class x h x w
#     :return:
#     '''
#     # tail_list = tail_list[0]
#     tail_list_rev = [None] * 20
#     for xx in range(14):
#         tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
#     tail_list_rev[14] = tail_list[15].unsqueeze(0)
#     tail_list_rev[15] = tail_list[14].unsqueeze(0)
#     tail_list_rev[16] = tail_list[17].unsqueeze(0)
#     tail_list_rev[17] = tail_list[16].unsqueeze(0)
#     tail_list_rev[18] = tail_list[19].unsqueeze(0)
#     tail_list_rev[19] = tail_list[18].unsqueeze(0)
#     return torch.cat(tail_list_rev,dim=0)


def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    n, h, w = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def get_parser():
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    LookupChoices = type('', (argparse.Action,), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--lr', default=1e-7, type=float)
    parser.add_argument('--numworker', default=12, type=int)
    parser.add_argument('--step', default=30, type=int)
    # parser.add_argument('--loadmodel',default=None,type=str)
    parser.add_argument('--classes', default=7, type=int)
    parser.add_argument('--loadmodel', default='', type=str)
    parser.add_argument('--txt_file', default='', type=str)
    parser.add_argument('--hidden_layers', default=128, type=int)
    parser.add_argument('--gpus', default=4, type=int)
    parser.add_argument('--input_path', default='./results/', type=str)
    parser.add_argument('--output_path', default='./results/', type=str)
    opts = parser.parse_args()
    return opts


def getModel(opts):
    backbone = 'xception' # Use xception or resnet as feature extractor,

    # Network definition
    if backbone == 'xception':
        net = deeplab_xception_transfer.deeplab_xception_transfer_projection(n_classes=opts.classes, os=16,
                                                                                     hidden_layers=opts.hidden_layers, source_classes=20,
                                                                                     )
    elif backbone == 'resnet':
        # net = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=7, os=16, pretrained=True)
        raise NotImplementedError
    else:
        raise NotImplementedError

    if gpu_id >= 0:
        net.cuda()

    # net load weights
    if not opts.loadmodel =='':
        x = torch.load(opts.loadmodel)
        net.load_source_model(x)
        print('load model:' ,opts.loadmodel)
    else:
        print('no model load !!!!!!!!')

    return net

def convertImg(imgNp):
    assert type(imgNp) == np.ndarray
    _img = Image.fromarray(imgNp)
    w, h = _img.size
    if w > 640 or h > 480:
        s = min(640 / w, 480 / h)
        w = int(s * w)
        h = int(s * h)
        _img = _img.resize((w, h))
    return _img



def inference(net, imgFile, opts):
    print('Processing file {}'.format(imgFile))
    im = Image.open(os.path.join(opts.input_path, imgFile))
    imgNp = np.array(im)

    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda()

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj1_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj3_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

    net.eval()

    img = convertImg(imgNp)

    ## multi scale
    scale_list=[1,0.5,0.75,1.25,1.5,1.75]
    testloader_list = []
    testloader_flip_list = []
    for pv in scale_list:
        composed_transforms_ts = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()])

        composed_transforms_ts_flip = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.HorizontalFlip_only_img(),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()])

        testloader_list.append(img_transform(img, composed_transforms_ts))
        testloader_flip_list.append(img_transform(img, composed_transforms_ts_flip))

    for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
        inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
        inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
        inputs = inputs.unsqueeze(0)
        inputs_f = inputs_f.unsqueeze(0)
        inputs = torch.cat((inputs, inputs_f), dim=0)
        if iii == 0:
            _, _, h, w = inputs.size()
        # assert inputs.size() == inputs_f.size()

        # Forward pass of the mini-batch
        inputs = Variable(inputs, requires_grad=False)

        with torch.no_grad():
            inputs = inputs.cuda()
            # outputs = net.forward(inputs)
            outputs = net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
            outputs = (outputs[0] + flip(outputs[1], dim=-1)) / 2
            outputs = outputs.unsqueeze(0)

            if iii > 0:
                outputs = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
                outputs_final = outputs_final + outputs
            else:
                outputs_final = outputs.clone()

    predictions = torch.max(outputs_final, 1)[1]
    prob_predictions = torch.max(outputs_final,1)[0]
    results = predictions.cpu().numpy()
    prob_results = prob_predictions.cpu().numpy()
    vis_res = decode_labels(results)

    if not os.path.isdir(opts.output_path):
        os.makedirs(opts.output_path)
    parsing_im = Image.fromarray(vis_res[0])
    parsing_im.save(opts.output_path + '/{}.vis.png'.format(imgFile.split('.')[0]))
    cv2.imwrite(opts.output_path + '/{}.png'.format(imgFile.split('.')[0]), results[0,:,:])


def main(opts):
    net = getModel(opts)

    with open(opts.txt_file, 'r') as f:
        img_list = f.readlines()

    for imgFile in img_list:
        imgFile = imgFile.strip()
        inference(net, imgFile, opts)


if __name__ == '__main__':
    opts = get_parser()
    main(opts)