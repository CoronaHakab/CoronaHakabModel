from enum import IntEnum, unique


@unique
class ConnectionTypes(IntEnum):
    Work = 1
    School = 2
    Family = 3
    Other = 4


# used for circles generation
In_Zone_types = [ConnectionTypes.School, ConnectionTypes.Family]
Multi_Zone_types = [ConnectionTypes.Work]
Whole_Population_types = [ConnectionTypes.Other]

# used for matrices generations
Connect_To_All_types = [ConnectionTypes.Family]
Random_Clustered_types = [ConnectionTypes.Work, ConnectionTypes.School]
Geographic_Clustered_types = [ConnectionTypes.Other]
