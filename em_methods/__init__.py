from logging.config import fileConfig
import os
from enum import Enum, auto

# Get module logger
base_path = os.path.dirname(os.path.abspath(__file__))
fileConfig(os.path.join(base_path, "logging.ini"),
           disable_existing_loggers=False)


class Units(Enum):
    M = auto()
    DM = auto()
    CM = auto()
    MM = auto()
    UM = auto()
    NM = auto()

    def convertSI(unit: 'Units') -> int:
        """
        Return convertion factor to SI (m) units
        m = 1, dm = 1e-1, cm = 1e-2, mm = 1e-3,
        um = 1e-6, nm = 1e-9
        """
        match unit:
            case Units.M:
                factor = 1
            case Units.DM:
                factor = 1e-1
            case Units.CM:
                factor = 1e-2
            case Units.MM:
                factor = 1e-3
            case Units.UM:
                factor = 1e-6
            case Units.NM:
                factor = 1e-9
        return factor

    def convertTo(self, unit2: 'Units') -> int:
        """ Convert unit to a different unit """
        return self.convertSI()/unit2.convertSI()

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name} ({Units.convertSI(self)})"

    def __str__(self) -> str:
        return f"{self.name.lower()} ({Units.convertSI(self)} m)"
    
    """ General function (Independent of enum definition) """

    def convertUnits(unit1: 'Units', unit2: 'Units') -> int:
        """ Convert from one unit to another"""
        return unit1.convertSI()/unit2.convertSI()

        
