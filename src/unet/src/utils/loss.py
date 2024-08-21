import torch
import torch.nn as nn

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'bce':
            return self.BCELoss
        else:
            raise NotImplementedError


    def BCELoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.BCELoss(reduction='mean')
        sigmoid = nn.Sigmoid()

        if self.cuda:
            criterion = criterion.cuda()

        logit = sigmoid(logit)
        # target = sigmoid(target)
        # print(logit.min(), logit.max())

        loss = 0

        for target_n2 in range(target.shape[1]):
            # temp_target = target[:, target_n2, :, :]
            # temp_logit = logit[:, target_n2, :, :]
            # target_t = [temp_target.min(), temp_target.max()]
            # logit_t = [temp_logit.min(), temp_logit.max()]
            loss_n2 = criterion(logit[:, target_n2, :, :], target[:, target_n2, :, :].float())
            loss += loss_n2

        # loss = criterion(logit, target.float())

        if self.batch_average:
            loss /= n

        return loss


    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




