from enum import IntEnum, unique


# the value represents the depth in the matrix
@unique
class ConnectionTypes(IntEnum):
    Work = 0
    Kindergarten = 1
    School = 2
    Family = 3
    Synagogue = 4
    Other = 5


# used for circles generation
In_Zone_types = [ConnectionTypes.School, ConnectionTypes.Kindergarten, ConnectionTypes.Family, ConnectionTypes.Synagogue]
Multi_Zone_types = [ConnectionTypes.Work]
Whole_Population_types = [ConnectionTypes.Other]
Non_Random_Age_Types = [ConnectionTypes.Family, ConnectionTypes.School]
Education_Types = [ConnectionTypes.School, ConnectionTypes.Kindergarten]

With_Random_Connections = [ConnectionTypes.Work, ConnectionTypes.School, ConnectionTypes.Kindergarten, ConnectionTypes.Synagogue, ConnectionTypes.Other]

# used for matrices generations
Connect_To_All_types = [ConnectionTypes.Family]
Random_Clustered_types = [ConnectionTypes.Work, ConnectionTypes.School, ConnectionTypes.Kindergarten, ConnectionTypes.School]
Geographic_Clustered_types = [ConnectionTypes.Other]
