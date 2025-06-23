import traci


class GetTraciData:
    """
    Main class combining all modules to interface with TRACI/SUMO.
    """

    def __init__(self):
        """Initialize submodules."""
        # Delayed imports to prevent circular dependencies
        from ACS.envs.common.sumo_interface import SUMOInterface
        from ACS.envs.common.vehicle_controller import VehicleController
        from ACS.envs.utils.state_utils import StateCalculator
        from ACS.envs.utils.collision_utils import CollisionDetector
        from ACS.envs.common.preheater import Preheater

        self.sumo = SUMOInterface()
        self.vehicle_controller = VehicleController()
        self.state_calculator = StateCalculator()
        self.collision_detector = CollisionDetector()
        self.preheater = Preheater()

        # Initialize base attributes
        self.av_id = self.vehicle_controller.av_id
        self.minspeed = self.vehicle_controller.minspeed
        self.maxspeed = self.vehicle_controller.maxspeed
        self.state_last = []
        self.step_length = self.sumo.step_length

        # Set detector references
        self.collision_detector.av_id = self.av_id
        self.vehicle_controller.av_id = self.av_id
        self.vehicle_controller.lane_width = self.sumo.lane_width
        self.state_calculator.av_id = self.av_id
        self.state_calculator.lane_width = self.sumo.lane_width

    def StartSimulation(self, Ifupdata=False, seed=1, env_type=0):
        self.sumo.start_simulation(Ifupdata, seed, env_type)

    def StepSimulation(self, action):
        self.vehicle_controller.control_vehicle(action, self.step_length)
        self.sumo.step_simulation()

    def GetState(self):
        return self.state_calculator.get_state(self.maxspeed)

    def GetVelocity(self, sv_id):
        return self.vehicle_controller.get_velocity_diff(sv_id)

    def CloseSimulation(self):
        self.sumo.close_simulation()

    def Preheat(self, time_length=0, lanechange_model_off=True):
        self.preheater.av_id = self.av_id
        self.preheater.step_length = self.step_length
        self.preheater.render_cmd = self.sumo.render_cmd
        self.preheater.lane_width = self.sumo.lane_width
        self.preheater.maxspeed = self.vehicle_controller.maxspeed

        self.preheater.preheat(time_length, lanechange_model_off)
        self.vehicle_controller.maxspeed = self.preheater.maxspeed  # Update maxspeed after preheat

    def CollisionDetection(self):
        self.collision_detector.av_id = self.av_id
        return self.collision_detector.check_collision()

    def GetTimeDone(self, simulation_time):
        current_time = traci.simulation.getTime()
        return self.collision_detector.check_time_done(current_time, self.preheat_time, simulation_time)

    def RuleModel(self):
        self.vehicle_controller.av_id = self.av_id
        self.vehicle_controller.maxspeed = self.maxspeed
        self.vehicle_controller.state_last = self.state_last
        return self.vehicle_controller.rule_model(self.state_last)


def GetTraciData_im():
    return GetTraciData()


if __name__ == '__main__':
    gtd = GetTraciData()
    gtd.StartSimulation()
    gtd.Preheat(57)
    gtd.GetState()
    gtd.CloseSimulation()