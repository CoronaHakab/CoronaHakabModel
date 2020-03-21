import yappi

from manager import SimulationManager


if __name__ == '__main__':
    #yappi.stop()
    sm = SimulationManager()
    exit()
    yappi.start()
    sm.run()