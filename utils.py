import os
from datetime import datetime
import json

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import _init_paths
from mylib.environ import PATH
from mylib.utils.misc import ConfigBase
from mylib.models.callbacks import CustomLogging

BASE = PATH.training_path


class TrainingBase(ConfigBase):

    flag = 'test'
    lr_schedule = dict(milestones=[30, 60, 100, 150],
                       values=[1e-3, 3e-4, 1e-4, 3e-5, 1e-5])

    def __init__(self, subfolder=True, **kwargs):

        super().__init__(**kwargs)

        self.starting = datetime.today().strftime("%y%m%d_%H%M%S")

        if subfolder:
            self._folder = os.path.join(BASE, self.flag, self.starting)
        else:
            self._folder = os.path.join(BASE, self.flag)

        if not os.path.exists(self._folder):
            os.makedirs(self._folder)

        with open(self._in_folder("config.json"), 'w') as f:
            json.dump(self._get_user_dict(serializable=True), f, indent=4)

    def _in_folder(self, path):
        return os.path.join(self._folder, path)

    def _lr_schedule(self, epoch):
        milestones = self.lr_schedule['milestones']
        values = self.lr_schedule['values']
        sel = 0
        for m in milestones:
            if epoch >= m:
                sel += 1
        sel_value = values[sel]
        print("Learning rate:", sel_value)
        return sel_value

    def _get_callbacks(self, **datasets):
        callbacks = []
        if self.lr_schedule:
            assert len(self.lr_schedule['milestones']) + 1 ==\
                len(self.lr_schedule['values'])
            lr_scheduler = LearningRateScheduler(self._lr_schedule, verbose=1)
            callbacks.append(lr_scheduler)
        checkpointer = ModelCheckpoint(filepath=self._in_folder('weights.{epoch:02d}.h5'),
                                       verbose=1, period=1, save_weights_only=True)
        custom_logging = CustomLogging(folder=self._folder, period=1,
                                       **datasets)
        return callbacks + [checkpointer, custom_logging]
