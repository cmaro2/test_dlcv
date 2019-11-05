import os
import torch

from parser_2 import arg_parse
import models

import data

import numpy as np
import torch.nn as nn

from tensorboardX import SummaryWriter
from test import evaluate


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':

    args = arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)


    val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='val'),
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=False)

    ''' load model '''
    print('===> prepare model ...')
    model = models.Net(args)
    """
    if os.path.isfile(os.path.join(args.save_dir, 'model_best.pth.tar')):
        model_std = torch.load(os.path.join(args.save_dir, 'model_best.pth.tar'))
        model.load_state_dict(model_std)
        """
    model.cuda()  # load model to gpu

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0
    start_epoch = 1

    for epoch in range(start_epoch, args.epoch + 1):

        model.train()

        for idx, (imgs, seg) in enumerate(train_loader):
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(train_loader))
            iters += 1

            ''' move data to gpu '''
            imgs, seg = imgs.cuda(), seg.cuda()

            ''' forward path '''
            output = model(imgs)

            #print(output.shape)

            ''' compute loss, backpropagation, update parameters '''
            loss = criterion(output, seg)  # compute loss

            optimizer.zero_grad()  # set grad of all parameters to zero
            loss.backward()  # compute gradient for each parameters
            optimizer.step()  # update parameters

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())


            print(train_info)

        if epoch % args.val_epoch == 0:
            ''' evaluate the model '''
            acc = evaluate(model, val_loader)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(model, os.path.join(args.save_dir, 'model_best.pth.tar'))
                best_acc = acc

        ''' save model '''
        save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))
