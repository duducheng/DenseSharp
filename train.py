from mylib.models.misc import set_gpu_usage
set_gpu_usage()

from mylib.dataloader.dataset import ClfSegDataset
from mylib.models import densenet
from utils import TrainingBase


class MyConfig(TrainingBase):

    flag = 'blending3_sgd_no_mixup'

    # model
    dhw = [48, 48, 48]
    down_structure = [3, 6, 4, 3]

    # training
    from keras.optimizers import SGD

    optimizer = SGD(lr=0, momentum=.9, nesterov=True)
    lr_schedule = dict(milestones=[30, 60, 100, 150],
                       values=[3e-2, 1e-2, 3e-3, 1e-3, 3e-4])
    n_pos = 20
    n_neg = 20
    weight_decay = 0.

    # data
    # alpha = 0.9  # for mixup
    alpha = None
    train_subset = [0, 1, 2]
    val_subset = [3]
    test_subset = [4]
    random_move = 8


def main(**kwargs):
    cfg = MyConfig(**kwargs)

    if cfg.alpha is None:
        train_dataset = ClfDataset(crop_size=cfg.dhw,
                                   subset=cfg.train_subset,
                                   move=cfg.random_move)
    loader = train_dataset.turn_to_resampling_loader(cfg.n_pos, cfg.n_neg)

    datasets = dict(train=train_dataset)
    if cfg.val_subset:
        datasets['val'] = ClfDataset(crop_size=cfg.dhw,
                                     subset=cfg.val_subset, move=None)
    if cfg.test_subset:
        datasets['test'] = ClfDataset(crop_size=cfg.dhw,
                                      subset=cfg.test_subset, move=None)

    callbacks = cfg._get_callbacks(**datasets)

    model = densenet.get_compiled(dhw=cfg.dhw,
                                  down_structure=cfg.down_structure,
                                  optimizer=cfg.optimizer,
                                  weight_decay=cfg.weight_decay,
                                  metrics=[])

    batch_size = cfg.n_pos + cfg.n_neg
    model.fit_generator(generator=loader, epochs=300,
                        steps_per_epoch=5 * len(train_dataset) // batch_size,
                        max_queue_size=500, workers=1,
                        callbacks=callbacks)


if __name__ == '__main__':

    import fire
    fire.Fire(main)
