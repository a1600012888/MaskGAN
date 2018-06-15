import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import argparse
from model import network
import torch.nn as nn

from critic import network as critic_network

from dataset import get_dataloaders

from cal_loss import vgg19, tv_loss, mask_l1, other_l1

from my_snip.clock import TrainClock, AvgMeter
from my_snip.config import save_args
from my_snip.torch_checkpoint import  save_checkpoint
from evaluate import test_on_benchmark, evaluate_on_val
from common import config
torch.backends.cudnn.benchmark = True

AdLossWeight = 3.0
ContentLossWeight = 10.0
TvLossWeight = 0.01
Valid_Loss_weight = 1
Hole_Loss_weight = 6

FakeLossWeight = 1.0
def train(generator, critic, optimizer_G, optimizer_C, data_loader,
          vgg_for_perceptual_loss, clock, writer, iter = 2):

    AdLosses = AvgMeter()
    TvLosses = AvgMeter()
    FakeLosses = AvgMeter()
    ContentLosses = AvgMeter()
    DLosses = AvgMeter()

    CriticLoss = nn.BCELoss()
    MaskLoss = mask_l1()
    OtherLoss = other_l1()
    TV_Loss = tv_loss()
    CriticLoss.cuda()
    MaskLoss.cuda()
    OtherLoss.cuda()
    TV_Loss.cuda()

    clock.tock()
    generator.eval()
    critic.train()
    pbar = tqdm(data_loader)
    for i, data in enumerate(pbar):
        clock.tick()


        inputs = data['hole_img'].float().cuda()
        labels = data['ori_img'].float().cuda()
        masks = data['mask'].float().cuda()

        outputs = generator(inputs)

        composite = torch.add(outputs * masks, labels * (1 - masks))

        critic_outputs = critic(composite.detach())

        d_loss = CriticLoss(critic_outputs ,masks)

        optimizer_C.zero_grad()
        d_loss.backward()

        optimizer_C.step()

        DLosses.update(d_loss.item())

        fake_loss = CriticLoss(critic(outputs), torch.ones_like(masks))
        FakeLosses.update(fake_loss.item())

        fake_loss = fake_loss * FakeLossWeight
        optimizer_C.zero_grad()
        fake_loss.backward()
        optimizer_C.step()

        if clock.minibatch % 200 == 1:
            print('Critic DL Loss :{:3f} -- FakeLoss: {:.3f}'.format(DLosses.mean, FakeLosses.mean))
        if clock.minibatch % iter == 1:

            critic.eval()
            generator.train()

            outputs = generator(inputs)# !!!???
            composite = torch.add(outputs * masks, labels * (1 - masks))

            critic_outputs = critic(composite)
            ad_loss = CriticLoss(critic_outputs, torch.zeros_like(masks)) # not taking other region into considerations

            perceptual_contents = vgg_for_perceptual_loss(outputs, labels).mean(dim=0)
            p_loss = torch.sum(perceptual_contents)

            tv_l = TV_Loss(outputs, labels, masks)
            hole_loss = MaskLoss(outputs, labels, masks)
            other_loss = OtherLoss(outputs, labels, masks)

            generator_loss = tv_l * TvLossWeight + hole_loss * Hole_Loss_weight + \
                other_loss * Valid_Loss_weight + p_loss * ContentLossWeight + \
                ad_loss * AdLossWeight

            #generator_loss = p_loss
            optimizer_G.zero_grad()

            generator_loss.backward()

            #print(generator.out_conv.weight.grad)
            optimizer_G.step()

            TvLosses.update(tv_l.item())
            ContentLosses.update(p_loss.item())
            AdLosses.update(ad_loss.item())
            generator.eval()
            critic.train()

            if (clock.minibatch // iter) % 100 == 1:
                print('Generator Loss: AdLoss:{:.3f} - TvLoss:{:.3f} - ContentLoss:{:3f}'.format(AdLosses.mean,
                                                                                                 TvLosses.mean,
                                                                                                 ContentLosses.mean))
        pbar.set_description("EPOCH[{}][{}/{}]".format(clock.epoch, i, len(data_loader)))
        pbar.set_postfix(
            content="{:.4f}".format(ContentLosses.mean))

    # After one epoch
    writer.add_scalar('Train/CriticLoss', DLosses.mean, clock.epoch)
    writer.add_scalar('Train/TvLoss', TvLosses.mean, clock.epoch)
    writer.add_scalar('Train/ContentLoss', ContentLosses.mean, clock.epoch)
    writer.add_scalar('Train/AdLoss', AdLosses.mean, clock.epoch)
    writer.add_scalar('Train/FakeLoss', FakeLosses.mean, clock.epoch)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=200, type=int, help='epoch number(Default:200)')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch number')
    parser.add_argument('-b', '--batch_size', default=6, type=int, help='mini-batch size(Default:6)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, help='initial learning rate(Default:1e-3)')
    parser.add_argument('--resume', type = str, default=None, help = 'The path for checkpoint file')
    parser.add_argument('--exp', type = str, default='test', help = 'The name of this exp')

    parser.add_argument('--content', type=float, default=10.0, help = 'the weight of content loss(Default:10.0)')
    parser.add_argument('--tv', type=float, default=1e-3, help = 'the weight of TV loss(Default:0.01)')
    parser.add_argument('--adv', type = float, default=3.0, help = 'the weight of adv loss(Default:3.0)')
    parser.add_argument('--fake', type=float, default=1.0, help='the weight of adv loss(Default:1.0)')

    args = parser.parse_args()
    base_dir = './exps/'
    exp_dir = os.path.join(base_dir, args.exp)
    base_results_dir = os.path.join(exp_dir, 'results/')
    best_metric = 0
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    if not os.path.exists(base_results_dir):
        os.mkdir(base_results_dir)

    save_args(args, exp_dir)

    global AdLossWeight
    global TvLossWeight
    global ContentLossWeight
    global FakeLossWeight
    AdLossWeight = args.adv
    TvLossWeight = args.tv
    ContentLossWeight = args.content
    FakeLossWeight = args.fake
    log_dir = os.path.join('./logs', args.exp)

    writer = SummaryWriter(log_dir)

    generator = network()
    critic = critic_network()
    optimizer_G = optim.Adam(generator.parameters(), args.lr)
    optimizer_C = optim.Adam(critic.parameters(), args.lr)
    scheduler_C = ReduceLROnPlateau(optimizer_G, 'min', factor=0.2, patience=10, verbose=True)
    scheduler_G = ReduceLROnPlateau(optimizer_C, 'min', factor=0.2, patience=10, verbose=True)

    if args.resume != None:
        assert os.path.exists(args.resume), 'model does not exist!'
        print('=> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        generator.load_state_dict(checkpoint['generator'])
        critic.load_state_dict(checkpoint['critic'])
        #best_metric = checkpoint['best_metric']
        optimizer_G = checkpoint['optimizer_G']
        optimizer_C = checkpoint['optimizer_C']

        print('=> loaded checkpoint {} - epoch:{} - best_metric:{}'.format(args.resume, args.start_epoch, best_metric))
    else:
        print('No checkpoint. A new begining')
    vgg_for_perceptual_loss = vgg19()
    for p in vgg_for_perceptual_loss.parameters():
        p.requires_grad = False


    generator.cuda()
    critic.cuda()
    vgg_for_perceptual_loss.cuda()
    vgg_for_perceptual_loss.eval()
    clock = TrainClock()
    clock.epoch = args.start_epoch
    data_dir = config.data_dir
    train_loader = get_dataloaders(os.path.join(data_dir, 'train.json'),
                                   batch_size=args.batch_size, shuffle=True)
    valid_loader = get_dataloaders(os.path.join(config.data_dir, 'val.json'),
                                   batch_size=args.batch_size, shuffle=True)
    print('Begin training')

    for epoch in range(args.start_epoch, args.epochs):

        results_dir = os.path.join(base_results_dir, '{}'.format(epoch))
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        train(generator, critic, optimizer_G, optimizer_C, train_loader,
            vgg_for_perceptual_loss, clock, writer, 2)

        save_checkpoint(
            {'epoch': clock.epoch,
             'generator': generator.state_dict(),
             'critic':critic.state_dict(),
             'optimizer_G': optimizer_G,
             'optimizer_C': optimizer_C,
             }, is_best=True,
            prefix=exp_dir
        )
        torch.cuda.empty_cache()
        test_on_benchmark(generator, results_dir)
        torch.cuda.empty_cache()
        CriticRealLoss, ContentLoss = evaluate_on_val(generator, critic, valid_loader, vgg_for_perceptual_loss,
                                                      clock, writer, os.path.join(exp_dir, 'valresults.txt'))
        scheduler_C.step(CriticRealLoss)
        scheduler_G.step(ContentLoss)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()