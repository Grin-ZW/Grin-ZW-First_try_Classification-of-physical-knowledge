"""进行模型的评估"""

import config
from model import  ImdbModel
from dataset2 import get_dataloader
import torch
import torch.nn.functional as F
import numpy as np


def eval():
    model = ImdbModel().to(config.device)
    model.load_state_dict(torch.load("./models/model.pkl"))

    loss_list = []  # 保存每一次的损失
    acc_list = []
    test_dataloader = get_dataloader(train=False)
    with torch.no_grad():
        for idx,(input,target) in enumerate(test_dataloader):
            input = input.to(config.device)
            target = target.to(config.device)
            output = model(input)
            loss = F.nll_loss(output,target)
            loss_list.append(loss.item())

            #准确率
            pred = output.max(dim=-1)[-1]
            acc_list.append(pred.eq(target).cpu().float().mean())
    print("loss mean:{},acc mean:{}".format(np.mean(loss_list),np.mean(acc_list)))



if __name__ == '__main__':
    eval()

