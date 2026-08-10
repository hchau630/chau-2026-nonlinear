from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class _CellType:
    sign: int
    prob: float
    _targets: tuple[str]

    @property
    def targets(self):
        return {CellType[k] for k in self._targets}


# V1 L2/3 statistics, probabilities approximately equal to Allen Institute's mouse V1 L2/3 model (2020 Billeh)
# (here we taken VIP to be synonymous with the broader class of Htr3a inhibitory neurons)
# TODO: Redesign API to allow for different animals, areas, and layers, loading constants from json/toml files
class CellType(_CellType, Enum):
    PYR = (1, 0.85, ("PYR", "PV", "SST", "VIP"))
    PV = (-1, 0.043, ("PYR", "PV", "SST", "VIP"))
    SST = (-1, 0.032, ("PYR", "PV", "VIP"))
    VIP = (-1, 0.075, ("SST",))
