from test import evaluate
import models
import torch
import data
from mean_iou_evaluate import mean_iou_score
import os
from parser_2 import arg_parse

if __name__ == '__main__':
    args = arg_parse()

    model = models.Net(args)
    model_std = torch.load(os.path.join(args.save_dir, 'model_best.pth.tar'))
    model.load_state_dict(model_std)
    model.cuda()

    val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='val'),
                                                 batch_size=args.train_batch,
                                                 num_workers=args.workers,
                                                 shuffle=False)

    acc = evaluate(model, val_loader)
    print(acc)





