#  Copyright (c) 2020 the original author or authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from functools import lru_cache
from typing import Tuple, Union

import numpy as np
import tensorflow as tf

from srcext.rude_carnie.model import inception_v3, get_checkpoint
from srcext.facenet.facenet import prewhiten

from src.services.imgtools.types import Array3D
from src.services.facescan import core
from src.services.dto import plugin_result

IMAGE_SIZE = 160 # TODO: change to ENV


@lru_cache(maxsize=2)
def _get_rude_carnie_model(type: str, labels: Tuple):
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        images = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
        logits = inception_v3(len(labels), images, 1, False)
        init = tf.global_variables_initializer()

        model_dir = f'/app/ml/srcext/rude_carnie/models/{type}'
        model_checkpoint_path, global_step = get_checkpoint(model_dir, None, 'checkpoint')

        saver = tf.train.Saver()
        saver.restore(sess, model_checkpoint_path)
        softmax_output = tf.nn.softmax(logits)

        def get_value(img: Array3D) -> Tuple[Union[str, Tuple], float]:
            img = np.expand_dims(prewhiten(img), 0)
            output = sess.run(softmax_output, feed_dict={images:img})[0]
            best_i = int(np.argmax(output))
            return labels[best_i], output[best_i]
        return get_value


class BaseGADetector(core.BasePlugin):
    LABELS: Tuple[str]
    dto_class: plugin_result.GenderDTO

    def __call__(self, face_img: Array3D) -> dict:
        model = _get_rude_carnie_model(self.name, self.LABELS)
        value, probability = model(face_img)
        return {self.name: value, f'{self.name}_probability': probability}


class AgeDetector(BaseGADetector):
    name = 'age'
    LABELS = ((0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100))

    def __call__(self, face_img: Array3D):
        model = _get_rude_carnie_model(self.name, self.LABELS)
        value, probability = model(face_img)
        return plugin_result.AgeDTO(age=value, age_probability=probability)

class GenderDetector(BaseGADetector):
    name = 'gender'
    LABELS = ('male', 'female')

    def __call__(self, face_img: Array3D):
        model = _get_rude_carnie_model(self.name, self.LABELS)
        value, probability = model(face_img)
        return plugin_result.GenderDTO(gender=value, gender_probability=probability)
