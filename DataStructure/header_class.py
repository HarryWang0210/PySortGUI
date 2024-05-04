from pydantic import BaseModel


class BaseHeader(BaseModel):
    ADC: int | float
    Bank: int
    Comment: str
    ElectrodeImpedance: float
    H5FileName: str
    H5Location: str
    H5Name: str
    HighCutOff: int | float
    HighCutOffOrder: int
    HighCutOffType: str
    ID: int
    LowCutOff: int | float
    LowCutOffOrder: int
    LowCutOffType: str
    MaxAnalogValue: int
    MaxDigValue: int
    MinAnalogValue: int
    MinDigValue: int
    Name: str
    NotchFilterFrequ: float
    NotchFilterOrder: int
    NotchFilterType: str
    NumRecords: int
    Pin: int
    SamplingFreq: int
    SigUnits: str
    Threshold: int | float
    TimeFirstPoint: int
    Type: str
