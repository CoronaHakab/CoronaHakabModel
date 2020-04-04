from enum import IntEnum, unique


@unique
class ConnectionTypes(IntEnum):
    Work = 1
    School = 2
    Family = 3
    Other = 4


In_Zone_types = [ConnectionTypes.School, ConnectionTypes.Family]
Multi_Zone_types = [ConnectionTypes.Work]
Whole_Population_types = [ConnectionTypes.Other]
