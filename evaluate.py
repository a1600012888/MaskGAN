import torch
import cv2
import numpy as np
import json
import os
from torchvision import transforms
import argparse
from model import network
from critic import network as critic_network
from common import config
from my_snip.clock import AvgMeter, TrainClock
from tqdm import tqdm
import math
from numpy.lib.stride_tricks import as_strided as ast
from cal_loss import vgg19, tv_loss, mask_l1, other_l1

from my_snip.clock import TrainClock, AvgMeter
from dataset import get_dataloaders
from common import config
import torch.nn as nn
BENCHMARK_JSON = '/home/zhangtianyuan/yjzhang/cvdl/cityscapes/data/benchmark.json'
def psnr(img1, img2):
    '''
    note: img should be in uint8
    '''
    mse = np.mean((img1 -img2) ** 2)

    if mse == 0:
        return 1e+6

    peak = 255.0 ** 2

    return 10 * math.log10(peak / mse)


def test_on_benchmark(net, save_dir):
    # test model on benchmarks
    # resize ori_img before input it to network
    input_shape = (512, 256)

    '''
    # load network
    net = None
    net = network()
    net.load_state_dict(torch.load(model_dir)['state_dict'])
    '''
    net.cuda()
    net.eval()

    # open json file to get metadata
    with open(BENCHMARK_JSON, 'r') as fp:
        meta_data = json.load(fp)

    psnrs = AvgMeter()

    # start predicting
    pbar = tqdm(range(len(meta_data)))
    for idx in pbar:
        data_info = meta_data[idx]
        ori_img = cv2.imread(data_info['ori_path'])
        ori_img = cv2.resize(ori_img, input_shape)

        cv2.imwrite(os.path.join(save_dir, data_info['filename'][:-4]+'ori.png'), ori_img)

        # if you want to use center crop, modify these part
        sx = data_info['sx']
        sy = data_info['sy']
        ex = data_info['ex']
        ey = data_info['ey']

        hole_img = np.copy(ori_img)
        hole_img[sy:ey, sx:ex, :] = 255

        cv2.imwrite(os.path.join(save_dir, data_info['filename'][:-4]+'hole.png'), hole_img)

        # scale all pixels to 0-1
        hole_img = hole_img * 1.0 / 255
        mask = np.zeros((ori_img.shape[0], ori_img.shape[1], 1))
        mask[sy:ey, sx:ex, :] = 1

        hole_img = transforms.ToTensor()(hole_img).float().to(config.device)


        with torch.no_grad():
            outputs = net(hole_img.unsqueeze_(0))

            out_img = outputs[0].detach()

            out_img = out_img / (torch.max(out_img) - torch.min(out_img)) * 255
            out_img = np.rollaxis(out_img.cpu().numpy(), 0, 3)

            psnr_val = psnr(ori_img.astype(np.uint8), out_img.astype(np.uint8))
            psnrs.update(psnr_val)

            cv2.imwrite(os.path.join(save_dir, data_info['filename'][:-4]+'out.png'), out_img)

        pbar.set_postfix(
            psnr=":{:.4f}".format(psnrs.mean),
        )

    # save metrics
    out = {
        'psnr': psnrs.mean,
    }
    with open(os.path.join(save_dir,'metrics.json'),'w') as fp:
        json.dump(out, fp)

    print('test is done')

def evaluate_on_val(generator, critic , data_loader, vgg_for_perceptual_loss, clock, writer = None, results_path = None):
    AdLosses = AvgMeter()
    TvLosses = AvgMeter()
    FakeLosses = AvgMeter()
    ContentLosses = AvgMeter()
    DLosses = AvgMeter()
    CriticRealLosses = AvgMeter()
    CriticLoss = nn.BCELoss()
    MaskLoss = mask_l1()
    OtherLoss = other_l1()
    TV_Loss = tv_loss()
    CriticLoss.cuda()
    MaskLoss.cuda()
    OtherLoss.cuda()
    TV_Loss.cuda()

    generator.eval()
    critic.eval()
    '''
    for p in generator.parameters():
        p.requires_grad =False
    for p in critic.parameters():
        p.requires_grad = False
    '''
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for i, data in enumerate(pbar):

            inputs = data['hole_img'].float().cuda()
            labels = data['ori_img'].float().cuda()
            masks = data['mask'].float().cuda()

            outputs = generator(inputs)

            composite = torch.add(outputs * masks, labels * (1 - masks))

            critic_outputs = critic(composite)

            d_loss = CriticLoss(critic_outputs ,masks)
            d_loss.detach_()

            critic_real_loss = CriticLoss(critic(labels), torch.zeros_like(masks))
            critic_real_loss.detach_()
            fakeloss = CriticLoss(critic(outputs), torch.ones_like(masks))

            CriticRealLosses.update(critic_real_loss.item())
            DLosses.update(d_loss.item())
            FakeLosses.update(fakeloss.item())
            ad_loss = CriticLoss(critic_outputs, torch.zeros_like(masks)) # not taking other region into considerations

            perceptual_contents = vgg_for_perceptual_loss(outputs, labels).mean(dim=0)
            p_loss = torch.sum(perceptual_contents)
            p_loss.detach_()

            tv_l = TV_Loss(outputs, labels, masks)
            hole_loss = MaskLoss(outputs, labels, masks)
            other_loss = OtherLoss(outputs, labels, masks)

            '''
            generator_loss = tv_l * TvLossWeight + hole_loss * Hole_Loss_weight + \
                other_loss * Valid_Loss_weight + p_loss * ContentLossWeight + \
                ad_loss * AdLossWeight
            '''

            TvLosses.update(tv_l.item())
            ContentLosses.update(p_loss.item())
            AdLosses.update(ad_loss.item())

            pbar.set_description("EPOCH[{}][{}/{}]".format(clock.epoch, i, len(data_loader)))
            pbar.set_postfix(
                content="{:.4f}".format(ContentLosses.mean))

        # After one epoch
        if writer is not None:
            writer.add_scalar('Val/CriticRealLoss', CriticRealLosses.mean, clock.epoch)
            writer.add_scalar('Val/CriticLoss',  DLosses.mean, clock.epoch)
            writer.add_scalar('Val/TvLoss', TvLosses.mean, clock.epoch)
            writer.add_scalar('Val/ContentLoss',ContentLosses.mean, clock.epoch)
            writer.add_scalar('Val/AdLoss',  AdLosses.mean, clock.epoch)
            writer.add_scalar('Val/FakeLoss',  FakeLosses.mean, clock.epoch)
        print('Evaluation Done!  CriticRealLoss:{:4f} - FakeLoss:{:.4f}- CriticLoss:{:.4f} - AdLoss:{:.3f} - ContentLoss:{:.3f} - TvLoss:{:.3f} '.format(
            CriticRealLosses.mean, FakeLosses.mean,  DLosses.mean, AdLosses.mean, ContentLosses.mean, TvLosses.mean
        ))

        if results_path:

            dic = {'epoch': clock.epoch,
                   'CriticRealLoss': CriticRealLosses.mean,
                   'DLloss': DLosses.mean,
                   'CriticLos': AdLosses.mean,
                   'ContentLoss':ContentLosses.mean,
                   'TvLoss': TvLosses.mean,
                   'FakeLoss': FakeLosses.mean}
            with open(results_path, 'a') as f:
                f.write('\n')
                json.dump(dic, f)
    return CriticRealLosses.mean, ContentLosses.mean

def test(resume_path):
    generator = network()
    critic = critic_network()
    print('=> loading checkpoint {}'.format(resume_path))
    checkpoint = torch.load(resume_path)
    generator.load_state_dict(checkpoint['generator'])
    critic.load_state_dict(checkpoint['critic'])

    generator.cuda()
    critic.cuda()
    valid_loader = get_dataloaders(os.path.join(config.data_dir, 'val.json'),
                                   batch_size=6, shuffle=True)
    print('Begin training')

    pvgg = vgg19()
    pvgg.eval()
    pvgg.cuda()
    evaluate_on_val(generator, critic, valid_loader, pvgg, TrainClock(), None, './test.txt')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_path', type=str, required=True, help='where to load the model')
    #parser.add_argument('--save_dir', type=str, required=True, help='where to save result img')
    args = parser.parse_args()

    '''
    net = network()
    net.load_state_dict(torch.load(args.resume_path)['generator'])

    if os.path.exists(args.save_dir) == False:
        os.mkdir(args.save_dir)

    test_on_benchmark(net, args.save_dir)

    '''
    test(args.resume_path)
