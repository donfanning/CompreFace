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

import os
from importlib.util import find_spec

modules_by_lib = {
    'tensorflow': ('facenet', 'rude_carnie'),
    'mexnet': ('insightface',)
}
modules_to_skip = []
for lib, modules in modules_by_lib.items():
    if find_spec(lib) is None:
        modules_to_skip.extend(modules)


def pytest_ignore_collect(path):
    _, last_path = os.path.split(path)
    for module in modules:
        if last_path.startswith(module):
            return True
