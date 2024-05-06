import logging
import os
from datetime import datetime

from pydantic import BaseModel, Field, computed_field, field_validator
from pydantic_core.core_schema import FieldValidationInfo

logger = logging.getLogger(__name__)


class FileHeader(BaseModel):
    FullFileName: str = Field(exclude=True)
    DateTime: datetime
    FileMajorVersion: int
    FileMinorVersion: int
    HeaderLength: int
    NumChannels: int
    RecordingSystem: str = 'Unknown'
    SHA1: str = ''
    ID: str = ''
    Comment: str = ''
    H5FileName: str = ''
    H5Location: str = ''
    H5Name: str = ''

    Type: str = 'File'
    FileSize: int = Field(default=0, validate_default=True)
    FileAtime: float = Field(default=.0, validate_default=True)
    FileCtime: float = Field(default=.0, validate_default=True)
    FileMtime: float = Field(default=.0, validate_default=True)

    # @field_validator('FullFileName', mode='before')
    # def parseFullFileName(cls, v, header: FieldValidationInfo):
    #     if os.path.isfile(v):
    #         return os.path.abspath(v)
    #     return v

    @computed_field
    @property
    def FilePath(self) -> str:
        return os.path.split(self.FullFileName)[0]

    @property
    def BaseName(self) -> str:
        return os.path.split(self.FullFileName)[1]

    @computed_field
    @property
    def FileName(self) -> str:
        return os.path.splitext(self.BaseName)[0]

    @computed_field
    @property
    def FileExt(self) -> str:
        return os.path.splitext(self.BaseName)[1]

    @field_validator('FileSize')
    def setFileSize(cls, v, header: FieldValidationInfo):
        if v > 0:
            return v

        file_name = header.data['FullFileName']
        if os.path.isfile(file_name):
            return os.path.getsize(file_name)
        logger.warn('Not able to set FileSize for the file')
        return 0

    @field_validator('FileAtime')
    def setFileAtime(cls, v, header: FieldValidationInfo):
        if v > 0:
            return v

        file_name = header.data['FullFileName']
        if os.path.isfile(file_name):
            return os.path.getatime(file_name)
        logger.warn('Not able to set FileAtime for the file')
        return .0

    @field_validator('FileCtime')
    def setFileCtime(cls, v, header: FieldValidationInfo):
        if v > 0:
            return v

        file_name = header.data['FullFileName']
        if os.path.isfile(file_name):
            return os.path.getctime(file_name)
        logger.warn('Not able to set FileCtime for the file')
        return .0

    @field_validator('FileMtime')
    def setFileMtime(cls, v, header: FieldValidationInfo):
        if v > 0:
            return v

        file_name = header.data['FullFileName']
        if os.path.isfile(file_name):
            return os.path.getmtime(file_name)
        logger.warn('Not able to set FileMtime for the file')
        return .0


class BaseHeader(BaseModel):
    ADC: int | float
    Bank: int
    ID: int
    Name: str
    SamplingFreq: int
    SigUnits: str

    Pin: int = 0
    ElectrodeImpedance: float = .0
    H5FileName: str = ''
    H5Location: str = ''
    H5Name: str = ''
    Comment: str = ''
    HighCutOff: int | float = 0
    HighCutOffOrder: int = 0
    HighCutOffType: str = ''
    LowCutOff: int | float = 0
    LowCutOffOrder: int = 0
    LowCutOffType: str = ''
    MaxAnalogValue: int = 0
    MaxDigValue: int = 0
    MinAnalogValue: int = 0
    MinDigValue: int = 0
    NotchFilterFrequ: float = .0
    NotchFilterOrder: int = 0
    NotchFilterType: str = ''
    NumRecords: int = 0
    Threshold: int | float = 0
    TimeFirstPoint: int = 0
    Type: str


class RawsHeader(BaseHeader):
    Type: str = 'Raws'


class EventsHeader(BaseHeader):
    Type: str = 'Events'
    NumUnits: int = 0

    @computed_field
    @property
    def NumEvents(self) -> int:
        return self.NumUnits


class SpikesHeader(BaseHeader):
    Type: str = 'Spikes'
    NumUnits: int = 0
    ReferenceID: int = -1
