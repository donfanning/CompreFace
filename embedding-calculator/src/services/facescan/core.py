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

from time import time
from abc import ABC, abstractmethod
from typing import List

from src.services.dto.bounding_box import BoundingBoxDTO
from src.services.dto import plugin_result
from src.services.imgtools.types import Array3D


class ExecutionContext:
    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time()

    def __float__(self):
        return float(self.end - self.start)


class BasePlugin(ABC):

    def __new__(cls):
        """
        Plugins might cache models in properties so it has to be Singleton.
        """
        if not hasattr(cls, 'instance'):
            cls.instance = super(BasePlugin, cls).__new__(cls)
        return cls.instance

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def backend(self) -> str:
        return self.__class__.__module__.rsplit('.', 1)[-1]

    @abstractmethod
    def __call__(self, face_img: Array3D) -> plugin_result.PluginResultDTO:
        raise NotImplementedError


class BaseFaceDetector(BasePlugin):
    name = 'detector'

    def __call__(self, img: Array3D, det_prob_threshold: float = None,
                    face_plugins: List[BasePlugin] = ()):
        """ Returns cropped and normalized faces."""
        faces = self._fetch_faces(img, det_prob_threshold)
        for face in faces:
            self._apply_face_plugins(face, face_plugins)
        return faces

    def _fetch_faces(self, img: Array3D, det_prob_threshold: float = None):
        start = time()
        boxes = self.find_faces(img, det_prob_threshold)
        return [
            plugin_result.FaceDTO(
                img=img, face_img=self.crop_face(img, box), box=box,
                execution_time={self.name: (time() - start) / len(boxes)}
            ) for box in boxes
        ]

    def _apply_face_plugins(self, face: plugin_result.FaceDTO,
                            plugins: List[BasePlugin]):
        for plugin in plugins:
            start = time()
            face._plugins_dto.append(plugin(face._face_img))
            face.execution_time[plugin.name] = time() - start

    @abstractmethod
    def find_faces(self, img: Array3D, det_prob_threshold: float = None) -> List[BoundingBoxDTO]:
        """ Find face bounding boxes, without calculating embeddings"""
        raise NotImplementedError

    @abstractmethod
    def crop_face(self, img: Array3D, box: BoundingBoxDTO) -> Array3D:
        """ Crop face by bounding box and resize/squish it """
        raise NotImplementedError


class BaseCalculator(BasePlugin):
    name = 'calculator'

    def __call__(self, face_img: Array3D):
        return plugin_result.EmbeddingDTO(
            embedding=self.calc_embedding(face_img)
        )

    @abstractmethod
    def calc_embedding(self, face_img: Array3D) -> Array3D:
        """ Calculate embedding of a given face """
        raise NotImplementedError
