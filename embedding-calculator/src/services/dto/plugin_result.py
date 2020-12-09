import attr
from typing import Tuple, List, Optional, Dict

from src.services.dto.bounding_box import BoundingBoxDTO
from src.services.dto.json_encodable import JSONEncodable
from src.services.imgtools.types import Array1D, Array3D


class PluginResultDTO(JSONEncodable):
    def to_json(self) -> dict:
        """ Serialize only public properties """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


@attr.s(auto_attribs=True, frozen=True)
class EmbeddingDTO(PluginResultDTO):
    embedding: Array1D


@attr.s(auto_attribs=True, frozen=True)
class GenderDTO(PluginResultDTO):
    gender: str
    gender_probability: float = attr.ib(converter=float, default=None)


@attr.s(auto_attribs=True, frozen=True)
class AgeDTO(PluginResultDTO):
    age: Tuple[int, int]
    age_probability: float = attr.ib(converter=float, default=None)


@attr.s(auto_attribs=True)
class FaceDTO(PluginResultDTO):
    box: BoundingBoxDTO
    _img: Optional[Array3D]
    _face_img: Optional[Array3D]
    _plugins_dto: List[PluginResultDTO] = attr.Factory(list)
    execution_time: Dict[str, float] = attr.Factory(dict)

    def to_json(self):
        data = super().to_json()
        for plugin_dto in self._plugins_dto:
            data.update(plugin_dto.to_json())
        return data
