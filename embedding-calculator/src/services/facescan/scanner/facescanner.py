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

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.services.dto.bounding_box import BoundingBoxDTO
from src.services.dto.scanned_face import ScannedFace
from src.services.imgtools.types import Array3D


class FaceScanner(ABC):
    ID = None

    def __init__(self):
        assert self.ID

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceScanner, cls).__new__(cls)
        return cls.instance

    @abstractmethod
    def scan(self, img: Array3D, det_prob_threshold: float = None) -> List[ScannedFace]:
        """ Find face bounding boxes and calculate embeddings"""
        raise NotImplementedError

    @abstractmethod
    def find_faces(self, img: Array3D, det_prob_threshold: float = None) -> List[BoundingBoxDTO]:
        """ Find face bounding boxes, without calculating embeddings"""
        raise NotImplementedError


class MockScanner(FaceScanner):
    ID = 'MockScanner'

    def scan(self, img: Array3D, det_prob_threshold: float = None) -> List[ScannedFace]:
        return [ScannedFace(box=BoundingBoxDTO(0, 0, 0, 0, 0), embedding=np.random.rand(1), img=img, face_img=img)]

    def find_faces(self, img: Array3D, det_prob_threshold: float = None) -> List[BoundingBoxDTO]:
        return [BoundingBoxDTO(0, 0, 0, 0, 0)]
