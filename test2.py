import os
import torch

import parser_2
import models_improved
import data2

from PIL import Image

import numpy as np
from mean_iou_evaluate import mean_iou_score

def evaluate(model, data_loader):
    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)

            _, pred = torch.max(pred, dim=1)

            pred = pred.cpu().numpy()
            #gt = gt.numpy().squeeze()

            preds.append(pred)
            gts.append(gt)

    #torchvision.utils.save_image(pred[0], 'image1.png')
    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    return mean_iou_score(preds, gts,9)


def save_imgs(model, images):
    model.eval()
    print("Saving Images in folder")
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (imgs, name) in enumerate(images):
            imgs = imgs.cuda()
            pred = model(imgs)
            #print("batch predicted")
            _, pred = torch.max(pred, dim=1)
            pred = pred.cpu().numpy()
            n=0
            dirs = []
            dirs.append(name)
            dirs = np.concatenate(dirs)
            for i in pred:
                result = Image.fromarray((i).astype(np.uint8))
                result.save(args.save_img + '/' + dirs[n])
                #torchvision.utils.save_image(i, 'preds/' + dirs[n])
                n += 1


if __name__ == '__main__':
    args = parser_2.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')

    test_loader2 = torch.utils.data.DataLoader(data2.DATA2(args),
                                              batch_size=args.test_batch,
                                              num_workers=args.workers,
                                              shuffle=False)

    ''' prepare mode '''
    model = models_improved.myModel()
    model_std = torch.load(os.path.join('model_best_impr.pth.tar'))
    model.load_state_dict(model_std)
    model.cuda()
    print("model loaded")

    save_imgs(model,test_loader2)

