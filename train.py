from mylib.models.misc import set_gpu_usage

set_gpu_usage()

from mylib.dataloader.dataset import ClfSegDataset, get_balanced_loader, get_loader
from mylib.models import densesharp, metrics, losses

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam


def main(batch_sizes, crop_size, random_move, learning_rate,
         segmentation_task_ratio, weight_decay, save_folder, epochs):
    '''

    :param batch_sizes: the number of examples of each class in a single batch
    :param crop_size: the input size
    :param random_move: the random move in data augmentation
    :param learning_rate: learning rate of the optimizer
    :param segmentation_task_ratio: the weight of segmentation loss in total loss
    :param weight_decay: l2 weight decay
    :param save_folder: where to save the snapshots, tensorflow logs, etc.
    :param epochs: how many epochs to run
    :return:
    '''
    batch_size = sum(batch_sizes)

    train_dataset = ClfSegDataset(crop_size=crop_size, subset=[0, 1, 2, 3], move=random_move,
                                  define_label=lambda l: [l[0] + l[1], l[2], l[3]])

    val_dataset = ClfSegDataset(crop_size=crop_size, subset=[4], move=None,
                                define_label=lambda l: [l[0] + l[1], l[2], l[3]])

    train_loader = get_balanced_loader(train_dataset, batch_sizes=batch_sizes)
    val_loader = get_loader(val_dataset, batch_size=batch_size)

    model = densesharp.get_compiled(output_size=3,
                                    optimizer=Adam(lr=learning_rate),
                                    loss={"clf": 'categorical_crossentropy',
                                          "seg": losses.DiceLoss()},
                                    metrics={'clf': ['accuracy', metrics.precision, metrics.recall, metrics.fmeasure,
                                                     metrics.invasion_acc, metrics.invasion_fmeasure,
                                                     metrics.invasion_precision, metrics.invasion_recall,
                                                     metrics.ia_acc, metrics.ia_fmeasure,
                                                     metrics.ia_precision, metrics.ia_recall],
                                             'seg': [metrics.precision, metrics.recall, metrics.fmeasure]},
                                    loss_weights={"clf": 1., "seg": segmentation_task_ratio},
                                    weight_decay=weight_decay)

    checkpointer = ModelCheckpoint(filepath='tmp/%s/weights.{epoch:02d}.h5' % save_folder, verbose=1,
                                   period=1, save_weights_only=True)
    best_keeper = ModelCheckpoint(filepath='tmp/%s/best.h5' % save_folder, verbose=1, save_weights_only=True,
                                  monitor='val_clf_acc', save_best_only=True, period=1, mode='max')
    csv_logger = CSVLogger('tmp/%s/training.csv' % save_folder)
    tensorboard = TensorBoard(log_dir='tmp/%s/logs/' % save_folder)
    early_stopping = EarlyStopping(monitor='val_clf_acc', min_delta=0, mode='max',
                                   patience=30, verbose=1)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.334, patience=10,
                                   verbose=1, mode='min', epsilon=1.e-5, cooldown=2, min_lr=0)

    model.fit_generator(generator=train_loader, steps_per_epoch=len(train_dataset), max_queue_size=500, workers=1,
                        validation_data=val_loader, epochs=epochs, validation_steps=len(val_dataset),
                        callbacks=[checkpointer, early_stopping, best_keeper, lr_reducer, csv_logger, tensorboard])


if __name__ == '__main__':
    main(batch_sizes=[3, 5, 8, 8],
         crop_size=[32, 32, 32],
         random_move=3,
         learning_rate=1.e-4,
         segmentation_task_ratio=0.2,
         weight_decay=0.,
         save_folder='test',
         epochs=100)
