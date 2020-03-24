from consts import Consts
from medical_state import (
    INFECTABLE_MEDICAL_STATES,
    INFECTIONS_MEDICAL_STATES,
    MedicalState,
)


class Agent:
    """
    This class represents a person in our doomed world.
    """

    __slots__ = (
        "ssn",
        "ID",
        "home",
        "work",
        "medical_state",
        "infection_date",
        "is_home_quarantined",
        "is_full_quarantined",
        "next_medical_state_transmition_date",
        "next_medical_state",
    )

    def __init__(self, ssn):
        self.ssn = ssn
        self.ID = ssn
        self.home = None
        self.work = None
        self.infection_date = None
        self.medical_state = MedicalState.Healthy
        self.is_home_quarantined = False
        self.is_full_quarantined = False
        self.next_medical_state_transmition_date = -1
        self.next_medical_state = MedicalState.Latent

    def __str__(self):
        return "<Person,  ssn={}, medical={}>".format(self.ssn, self.medical_state)

    def is_infectious(self):
        """
        Check if this agent is infectious.

        :return: bool, True if this agent can infect others.
        """

        # pay attantion, doesn't have to be only in this stage. in the future this could be multiple stages check
        return self.medical_state in INFECTIONS_MEDICAL_STATES

    def is_infectable(self):
        return self.medical_state in INFECTABLE_MEDICAL_STATES

    def infect(self, date=0, manager=None):
        """
        Will try to infect this agent with given probability
        """
        if self.is_infectable():
            self.change_medical_state(current_date=date, manager=manager)
            self.infection_date = date
            return True

    def get_infection_ratio(self, consts: Consts):
        if self.medical_state == MedicalState.Symptomatic:
            return consts.Symptomatic_infection_ratio
        elif self.medical_state == MedicalState.Asymptomatic:
            return consts.ASymptomatic_infection_ratio
        elif self.medical_state == MedicalState.Silent:
            return consts.Silent_infection_ratio
        # right now you are not infecting when in hospital
        # elif self.medical_state == MedicalState.Hospitalized:
        #    return consts.Symptomatic_infection_ratio # todo find out the real number
        # elif self.medical_state == MedicalState.Icu:
        #    return consts.Symptomatic_infection_ratio # todo find out the real number
        return 0

    def add_home(self, home):
        self.home = home

    def add_work(self, work):
        self.work = work

    def change_medical_state(self, current_date: int, roll=-1, manager=None):
        # todo now doesnt roll for next transition date
        if current_date < self.next_medical_state_transmition_date:
            return "nothing new"
        else:
            if manager is not None:
                if self.medical_state != MedicalState.Healthy:
                    manager.counters[self.medical_state] = (
                        manager.counters[self.medical_state] - 1
                    )
                manager.counters[self.next_medical_state] = (
                    manager.counters[self.next_medical_state] + 1
                )
            if self.next_medical_state == MedicalState.Latent:
                self.medical_state = MedicalState.Latent
                self.next_medical_state = MedicalState.Silent
                self.next_medical_state_transmition_date = (
                    current_date + Consts.average_latent_to_silent_days
                )
                return "new latent"
            elif self.next_medical_state == MedicalState.Silent:
                self.medical_state = MedicalState.Silent
                if (
                    roll < Consts.silent_to_asymptomatic_probability
                ):  # next is asymptomatic
                    self.next_medical_state = MedicalState.Asymptomatic
                    self.next_medical_state_transmition_date = (
                        current_date + Consts.average_silent_to_asymptomatic_days
                    )
                else:
                    self.next_medical_state = MedicalState.Symptomatic
                    self.next_medical_state_transmition_date = (
                        current_date + Consts.average_silent_to_symptomatic_days
                    )
                return "new_infecting"
            elif self.next_medical_state == MedicalState.Asymptomatic:
                self.medical_state = MedicalState.Asymptomatic
                self.next_medical_state = MedicalState.Immune
                self.next_medical_state_transmition_date = (
                    current_date + Consts.average_asymptomatic_to_recovered_days
                )
                return "new asymptomatic"
            elif self.next_medical_state == MedicalState.Immune:
                self.medical_state = MedicalState.Immune
                self.next_medical_state_transmition_date = -1
                return "new_not_infecting"
            elif self.next_medical_state == MedicalState.Symptomatic:
                self.medical_state = MedicalState.Symptomatic
                if roll < Consts.symptomatic_to_asymptomatic_probability:
                    self.next_medical_state = MedicalState.Asymptomatic
                    self.next_medical_state_transmition_date = (
                        current_date + Consts.average_symptomatic_to_asymptomatic_days
                    )
                else:
                    self.next_medical_state = MedicalState.Hospitalized
                    self.next_medical_state_transmition_date = (
                        current_date + Consts.average_symptomatic_to_hospitalized_days
                    )
                    return "new symptomatic"
            elif self.next_medical_state == MedicalState.Hospitalized:
                self.medical_state = MedicalState.Hospitalized
                if roll < Consts.hospitalized_to_asymptomatic_probability:
                    self.next_medical_state = MedicalState.Asymptomatic
                    self.next_medical_state_transmition_date = (
                        current_date + Consts.average_hospitalized_to_asymptomatic_days
                    )
                else:
                    self.next_medical_state = MedicalState.Icu
                    self.next_medical_state_transmition_date = (
                        current_date + Consts.average_hospitalized_to_icu_days
                    )
                    return "new hospitalized"
            elif self.next_medical_state == MedicalState.Icu:
                self.medical_state = MedicalState.Icu
                if roll < Consts.icu_to_hospitalized_probability:
                    self.next_medical_state = MedicalState.Hospitalized
                    self.next_medical_state_transmition_date = (
                        current_date + Consts.average_icu_to_hospitalized_days
                    )
                else:
                    self.next_medical_state = MedicalState.Deceased
                    self.next_medical_state_transmition_date = (
                        current_date + Consts.average_icu_to_dead_days
                    )
                    return "new icu"
            elif self.next_medical_state == MedicalState.Deceased:
                self.medical_state = MedicalState.Deceased
                return "new_not_infecting"

        return "nothing new"

    def __cmp__(self, other):
        return self.ID == other.ID


class Circle:
    __slots__ = "type", "agents"

    def __init__(self, type):
        self.type = type
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def get_indexes_of_my_circle(self, my_index):
        rest_of_circle = {o.ID for o in self.agents}
        rest_of_circle.remove(my_index)
        return rest_of_circle
