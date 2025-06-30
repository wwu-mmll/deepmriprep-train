from monai.losses import GeneralizedDiceLoss
from fastai.basics import F, Tensor, TensorBase, FocalLoss


class DiceFocalLoss:
    def __init__(self, cls_props=None, lambda_focal=.2, lambda_gdl=1., beta=.999, gamma=2., include_background=True):
        weight = self.loss_weights(cls_props, beta)
        weight = Tensor(weight).cuda()
        self.focal_func = FocalLoss(gamma=gamma, weight=weight)
        self.dice_func = GeneralizedDiceLoss(include_background, softmax=False, to_onehot_y=True)
        self.lambda_focal, self.lambda_gdl = lambda_focal, lambda_gdl

    def __call__(self, pred: Tensor, targ: Tensor) -> Tensor:
        pred, targ = TensorBase(pred), TensorBase(targ)
        pred = F.softmax(pred, dim=1)
        return self.lambda_focal * self.focal_func(pred, targ.long()) + self.lambda_gdl * self.dice_func(pred, targ[None])

    def activation(self, x):
        return F.softmax(x, dim=1).type(x.dtype)

    @staticmethod
    def loss_weights(cls_props, beta):
        if cls_props is None:
            return None
        else:
            weight = (1 - beta) / (1 - beta ** Tensor(cls_props).double())
            weight = weight.float()
            weight /= weight.mean()
            return weight.tolist()
