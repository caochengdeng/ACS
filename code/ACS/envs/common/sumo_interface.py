import os
import sys
import traci


class SUMOInterface:

    def __init__(self):
        self.sumocfgfile = r"Z:\codes\pythoncode\paper3\sumo_python_AIE\sumo\route_confi.sumocfg"
        self.render_cmd = "sumo"
        self.step_length = "0.1"
        self.lateral_resolution = "36"

    def start_simulation(self, Ifupdata=False, seed=1, env_type=0):

        if env_type == 1:
            self.sumocfgfile = r"./"
        elif env_type == 2:
            self.sumocfgfile = r"./"
        elif env_type == 3:
            self.sumocfgfile = r"./"

        if Ifupdata:
            traci.start([
                self.render_cmd, "--step-length", self.step_length, "--lateral-resolution", "6",
                "-c", self.sumocfgfile, "--seed", str(seed)
            ])
        else:
            traci.start([
                self.render_cmd, "--step-length", self.step_length, "--lateral-resolution", "6",
                "-c", self.sumocfgfile, "--random"
            ])

    def step_simulation(self):

        traci.simulationStep()

    def close_simulation(self):
        """
        Close the current simulation.
        """
        traci.close()

    @staticmethod
    def check_sumo_home():
        if 'SUMO_HOME' not in os.environ:
            sys.exit("Please declare environment variable 'SUMO_HOME'")