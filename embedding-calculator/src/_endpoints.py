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
from typing import List, Optional

from flask import request
from flask.json import jsonify
from werkzeug.exceptions import BadRequest

from src.constants import ENV
from src.exceptions import NoFaceFoundError
from src.services.facescan import helpers
from src.services.facescan.scanner import facescanner as scanners
from src.services.facescan.scanner.facescanners import scanner
from src.services.flask_.constants import ARG
from src.services.flask_.needs_attached_file import needs_attached_file
from src.services.imgtools.read_img import read_img


def endpoints(app):
    @app.route('/status')
    def status_get():
        availiable_plugins = {p.name: p.backend for p in helpers.get_plugins()}
        return jsonify(status='OK', build_version=ENV.BUILD_VERSION,
                       calculator_version=scanner.ID,
                       availiable_plugins=availiable_plugins)

    @app.route('/find_faces', methods=['POST'])
    @needs_attached_file
    def find_faces_post():
        detector = helpers.get_detector(_get_det_plugin_name())
        face_plugins = helpers.get_plugins(_get_face_plugin_names())

        faces = detector(
            img=read_img(request.files['file']),
            det_prob_threshold=_get_det_prob_threshold(),
            face_plugins=face_plugins
        )
        plugins_versions = {p.name: p.backend for p in [detector] + face_plugins}
        return jsonify(results=faces, plugins_versions=plugins_versions)

    @app.route('/scan_faces', methods=['POST'])
    @needs_attached_file
    def scan_faces_post():
        faces = scanner.scan(
            img=read_img(request.files['file']),
            det_prob_threshold=_get_det_prob_threshold()
        )
        faces = _limit(faces, request.values.get(ARG.LIMIT))
        return jsonify(calculator_version=scanner.ID, result=faces)


def _get_det_prob_threshold():
    det_prob_threshold_val = request.values.get(ARG.DET_PROB_THRESHOLD)
    if det_prob_threshold_val is None:
        return None
    det_prob_threshold = float(det_prob_threshold_val)
    if not (0 <= det_prob_threshold <= 1):
        raise BadRequest('Detection threshold incorrect (0 <= det_prob_threshold <= 1)')
    return det_prob_threshold


def _get_det_plugin_name() -> Optional[str]:
    if ARG.DET_PLUGIN not in request.values:
        return
    return request.values.get(ARG.DET_PLUGIN)


def _get_face_plugin_names() -> Optional[List[str]]:
    if ARG.FACE_PLUGINS not in request.values:
        return
    return [
        name for name in request.values[ARG.FACE_PLUGINS].split(',') if name
    ]


def _limit(faces: List, limit: str = None) -> List:
    """
    >>> _limit([1, 2, 3], None)
    [1, 2, 3]
    >>> _limit([1, 2, 3], '')
    [1, 2, 3]
    >>> _limit([1, 2, 3], 0)
    [1, 2, 3]
    >>> _limit([1, 2, 3], 1)
    [1]
    >>> _limit([1, 2, 3], 2)
    [1, 2]
    """
    if len(faces) == 0:
        raise NoFaceFoundError

    try:
        limit = int(limit or 0)
    except ValueError as e:
        raise BadRequest('Limit format is invalid (limit >= 0)') from e
    if not (limit >= 0):
        raise BadRequest('Limit value is invalid (limit >= 0)')

    return faces[:limit] if limit else faces
