import logging
import os
from dataclasses import dataclass, field
from datetime import datetime

from .DataClasses import DataClass, convert_and_enforce_types

# from pydantic import BaseModel, Field, computed_field, field_validator
# from pydantic_core.core_schema import FieldValidationInfo

logger = logging.getLogger(__name__)


@convert_and_enforce_types(convert_types=[(datetime, lambda value: datetime.strptime(value.decode(), "%Y-%m-%d %H:%M:%S"))])
@dataclass
class FileHeader(DataClass):
    DateTime: datetime
    FileMajorVersion: int
    FileMinorVersion: int
    HeaderLength: int
    NumChannels: int
    FullFileName: str = ''
    FilePath: str = ''
    # BaseName: str = field(default='', init=False)
    FileName: str = ''
    FileExt: str = ''

    RecordingSystem: str = 'Unknown'
    SHA1: str = ''
    ID: str = ''
    Comment: str = ''
    H5FileName: str = ''
    H5Location: str = ''
    H5Name: str = ''

    Type: str = 'File'
    FileSize: int = 0
    FileAtime: float = .0
    FileCtime: float = .0
    FileMtime: float = .0

    def __post_init__(self):
        self._addIgnoreField('FullFileName')
        self.FilePath, self.BaseName = os.path.split(self.FullFileName)
        self.FileName, self.FileExt = os.path.splitext(self.BaseName)
        self._setFileSize()
        self._setFileAtime()
        self._setFileCtime()
        self._setFileMtime()

    def _setFileSize(self):
        if self.FileSize > 0:
            return

        if os.path.isfile(self.FullFileName):
            self.FileSize = os.path.getsize(self.FullFileName)
            return

        logger.warning('Not able to set FileSize for the file')

    def _setFileAtime(self):
        if self.FileAtime > 0:
            return

        if os.path.isfile(self.FullFileName):
            self.FileAtime = os.path.getatime(self.FullFileName)
            return

        logger.warning('Not able to set FileAtime for the file')

    def _setFileCtime(self):
        if self.FileCtime > 0:
            return

        if os.path.isfile(self.FullFileName):
            self.FileCtime = os.path.getctime(self.FullFileName)
            return
        logger.warning('Not able to set FileCtime for the file')

    def _setFileMtime(self):
        if self.FileMtime > 0:
            return

        if os.path.isfile(self.FullFileName):
            self.FileMtime = os.path.getmtime(self.FullFileName)
            return
        logger.warning('Not able to set FileMtime for the file')

    # def model_dump(self, *args, **kwargs):
    #     result = super().model_dump(*args, **kwargs)
    #     result.pop('FullFileName')
    #     return result


@dataclass
class BaseHeader(DataClass):
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
    Type: str = ''


@convert_and_enforce_types()
@dataclass
class RawsHeader(BaseHeader):
    Type: str = 'Raws'


@convert_and_enforce_types()
@dataclass
class EventsHeader(BaseHeader):
    Type: str = 'Events'
    NumUnits: int = 0
    NumEvents: int = 0

    def __post_init__(self):
        if self.NumEvents <= 0:
            self.NumEvents = self.NumUnits


@convert_and_enforce_types()
@dataclass
class SpikesHeader(BaseHeader):
    Type: str = 'Spikes'
    NumUnits: int = 0
    ReferenceID: int = -1
    Label: str = 'default'


# class FileHeader(BaseModel):
#     FullFileName: str = Field(default='', exclude=True)
#     DateTime: datetime
#     FileMajorVersion: int
#     FileMinorVersion: int
#     HeaderLength: int
#     NumChannels: int
#     RecordingSystem: str = 'Unknown'
#     SHA1: str = ''
#     ID: str = ''
#     Comment: str = ''
#     H5FileName: str = ''
#     H5Location: str = ''
#     H5Name: str = ''

#     Type: str = 'File'
#     FileSize: int = Field(default=0, validate_default=True)
#     FileAtime: float = Field(default=.0, validate_default=True)
#     FileCtime: float = Field(default=.0, validate_default=True)
#     FileMtime: float = Field(default=.0, validate_default=True)

#     # @field_validator('FullFileName', mode='before')
#     # def parseFullFileName(cls, v, header: FieldValidationInfo):
#     #     if os.path.isfile(v):
#     #         return os.path.abspath(v)
#     #     return v

#     @computed_field
#     @property
#     def FilePath(self) -> str:
#         return os.path.split(self.FullFileName)[0]

#     @property
#     def BaseName(self) -> str:
#         return os.path.split(self.FullFileName)[1]

#     @computed_field
#     @property
#     def FileName(self) -> str:
#         return os.path.splitext(self.BaseName)[0]

#     @computed_field
#     @property
#     def FileExt(self) -> str:
#         return os.path.splitext(self.BaseName)[1]

#     @field_validator('FileSize')
#     def setFileSize(cls, v, header: FieldValidationInfo):
#         if v > 0:
#             return v

#         file_name = header.data['FullFileName']
#         if os.path.isfile(file_name):
#             return os.path.getsize(file_name)
#         logger.warning('Not able to set FileSize for the file')
#         return 0

#     @field_validator('FileAtime')
#     def setFileAtime(cls, v, header: FieldValidationInfo):
#         if v > 0:
#             return v

#         file_name = header.data['FullFileName']
#         if os.path.isfile(file_name):
#             return os.path.getatime(file_name)
#         logger.warning('Not able to set FileAtime for the file')
#         return .0

#     @field_validator('FileCtime')
#     def setFileCtime(cls, v, header: FieldValidationInfo):
#         if v > 0:
#             return v

#         file_name = header.data['FullFileName']
#         if os.path.isfile(file_name):
#             return os.path.getctime(file_name)
#         logger.warning('Not able to set FileCtime for the file')
#         return .0

#     @field_validator('FileMtime')
#     def setFileMtime(cls, v, header: FieldValidationInfo):
#         if v > 0:
#             return v

#         file_name = header.data['FullFileName']
#         if os.path.isfile(file_name):
#             return os.path.getmtime(file_name)
#         logger.warning('Not able to set FileMtime for the file')
#         return .0


# class BaseHeader(BaseModel):
#     ADC: int | float
#     Bank: int
#     ID: int
#     Name: str
#     SamplingFreq: int
#     SigUnits: str

#     Pin: int = 0
#     ElectrodeImpedance: float = .0
#     H5FileName: str = ''
#     H5Location: str = ''
#     H5Name: str = ''
#     Comment: str = ''
#     HighCutOff: int | float = 0
#     HighCutOffOrder: int = 0
#     HighCutOffType: str = ''
#     LowCutOff: int | float = 0
#     LowCutOffOrder: int = 0
#     LowCutOffType: str = ''
#     MaxAnalogValue: int = 0
#     MaxDigValue: int = 0
#     MinAnalogValue: int = 0
#     MinDigValue: int = 0
#     NotchFilterFrequ: float = .0
#     NotchFilterOrder: int = 0
#     NotchFilterType: str = ''
#     NumRecords: int = 0
#     Threshold: int | float = 0
#     TimeFirstPoint: int = 0
#     Type: str


# class RawsHeader(BaseHeader):
#     Type: str = 'Raws'


# class EventsHeader(BaseHeader):
#     Type: str = 'Events'
#     NumUnits: int = 0

#     @computed_field
#     @property
#     def NumEvents(self) -> int:
#         return self.NumUnits


# class SpikesHeader(BaseHeader):
#     Type: str = 'Spikes'
#     NumUnits: int = 0
#     ReferenceID: int = -1
#     Label: str = 'default'
