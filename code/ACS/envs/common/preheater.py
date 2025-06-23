import traci


class Preheater:

    def __init__(self, av_id='DRL_AV'):
        self.av_id = av_id
        self.preheat_time = 0
        self.step_length = "0.1"

    def preheat(self, time_length=0, lanechange_model_off=True):
        """
        Run simulation without control for warm-up.
        """
        traci.simulationStep(time_length)
        self.preheat_time = time_length

        while self.av_id not in traci.vehicle.getIDList():
            traci.simulationStep()
            self.preheat_time += float(self.step_length)

        if lanechange_model_off:
            traci.vehicle.setLaneChangeMode(self.av_id, 0b000000000000)
        traci.vehicle.setSpeedMode(self.av_id, 0b00000)

        self.maxspeed = traci.vehicle.getMaxSpeed(self.av_id) * 3.6
        if self.render_cmd == "sumo-gui":
            traci.gui.trackVehicle("View #0", self.av_id)

        return self.preheat_time