from enum import IntEnum, unique


# the value represents the depth in the matrix
@unique
class ConnectionTypes(IntEnum):
    Work = 0
    Kindergarten = 1
    School = 2
    Family = 3
    Other = 4


# used for circles generation
In_Zone_types = [ConnectionTypes.School, ConnectionTypes.Kindergarten, ConnectionTypes.Family]
Multi_Zone_types = [ConnectionTypes.Work]
Whole_Population_types = [ConnectionTypes.Other]
Non_Random_Age_Types = [ConnectionTypes.Family, ConnectionTypes.School]
Education_Types = [ConnectionTypes.School, ConnectionTypes.Kindergarten]
Non_Exclusive_Types = [ConnectionTypes.Family, ConnectionTypes.Other]  # More precisely, not in [school, kindergarten, work]

With_Random_Connections = [ConnectionTypes.School, ConnectionTypes.Kindergarten, ConnectionTypes.Other]
With_Geo_Random_Connections = [ConnectionTypes.Work]

# used for matrices generations
Connect_To_All_types = [ConnectionTypes.Family]
Random_Clustered_types = [ConnectionTypes.Work, ConnectionTypes.School, ConnectionTypes.Kindergarten]
Geographic_Clustered_types = [ConnectionTypes.Other]
