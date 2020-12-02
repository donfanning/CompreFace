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

from importlib import import_module
from typing import List, Dict, Type

from src import constants, exceptions
from src.services.facescan.core import BasePlugin, BaseFaceDetector
from src.services.imgtools.types import Array3D


def import_classes(class_path: str):
    module, class_name = class_path.rsplit('.', 1)
    return getattr(import_module(module, __package__), class_name)


def import_plugins(paths: List[str]) -> List[Type[BasePlugin]]:
    return [import_classes(path) for path in paths]


def get_plugins(names: List[str] = None) -> List[BasePlugin]:
    return [fp() for fp in import_plugins(constants.FACE_PLUGINS)
            if names is None or fp.name in names]


def get_detector(name: str = None) -> BaseFaceDetector:
    for face_detector in import_plugins(constants.FACE_DETECTORS):
        if name is not None and face_detector.name != name:
            continue
        return face_detector()
    raise exceptions.InvalidFaceDetectorPlugin


class Scanner:
    """
    Class for backward compatibility.
    The scanner does only detection and embedding calculation.
    """
    ID = "PluginScanner"

    def scan(self, img: Array3D, det_prob_threshold: float = None):
        detector: BaseFaceDetector = get_detector()
        face_plugins = get_plugins(['calculator'])
        return detector(img, det_prob_threshold, face_plugins)
