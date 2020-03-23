from enum import IntEnum


class MedicalState(IntEnum):
    Immune = -1
    Healthy = 0
    Silent = 2
    Asymptomatic = 3
    Symptomatic = 4
    Deceased = 5


INFECTABLE_MEDICAL_STATES = {MedicalState.Healthy}
INFECTIONS_MEDICAL_STATES = {MedicalState.Asymptomatic, MedicalState.Symptomatic}
IMUNE_MEDICAL_STATES = {MedicalState.Immune, MedicalState.Deceased}