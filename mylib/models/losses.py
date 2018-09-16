import keras.backend as K


class DiceLoss:
    def __init__(self, beta=1., smooth=1.):
        self.__name__ = 'dice_loss_' + str(int(beta * 100))
        self.beta = beta  # the more beta, the more recall
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        bb = self.beta * self.beta
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f, axis=-1)
        weighted_union = bb * K.sum(y_true_f, axis=-1) + \
                         K.sum(y_pred_f, axis=-1)
        score = -((1 + bb) * intersection + self.smooth) / \
                (weighted_union + self.smooth)
        return score
