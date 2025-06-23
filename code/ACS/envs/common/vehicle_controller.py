import traci


class VehicleController:
    """
    Class to handle vehicle control actions such as lane change and acceleration.
    """

    def __init__(self):
        self.av_id = 'DRL_AV'
        self.minspeed = 80  # km/h
        self.maxspeed = 120  # km/h (will be updated later)
        self.lane_width = [3.6, 3.6, 3.6]

    def control_vehicle(self, action, step_length="0.1"):
        """
        Control the autonomous vehicle based on the provided action.
        """
        LaneId = traci.vehicle.getLaneIndex(self.av_id)

        if action[0] is not None:
            LaneID_Target = max(0, min(2, LaneId + action[0]))
            LaneChange_Distance = (LaneID_Target - LaneId) * (
                self.lane_width[LaneId] + self.lane_width[LaneID_Target]
            ) / 2 - traci.vehicle.getLateralLanePosition(self.av_id)
            traci.vehicle.changeSublane(self.av_id, LaneChange_Distance)

        speed_av = traci.vehicle.getSpeed(self.av_id)
        action_speed = action[1]

        if speed_av + float(step_length) * action[1] < self.minspeed / 3.6:
            action_speed = (-speed_av + self.minspeed / 3.6) / float(step_length)
        elif speed_av + float(step_length) * action[1] > self.maxspeed / 3.6:
            action_speed = (self.maxspeed / 3.6 - speed_av) / float(step_length)

        traci.vehicle.setAcceleration(self.av_id, 2 * action_speed, 0.1)

    def get_velocity_diff(self, sv_id):
        """
        Get velocity difference between AV and surrounding vehicles.
        """
        velocity = traci.vehicle.getSpeed(sv_id)
        velocity_av = traci.vehicle.getSpeed(self.av_id)
        return velocity_av - velocity

    def rule_model(self, state_last):
        """
        Rule-based model using IDM for decision making.
        """
        a_max, d_min, T, b, v = 3, 2, 1.1, 3, state_last[0]
        delta_v = state_last[5]
        distance_expect = d_min + T * v + v * delta_v / (2 * (a_max * b) ** 0.5)

        action_rule = [1, 0]
        action_rule[1] = a_max * (1 - (v * 3.6 / self.maxspeed) ** 4 - (distance_expect / state_last[3]) ** 2)
        action_rule[1] = action_rule[1] + 3 if -3 < action_rule[1] < 3 else 0 if action_rule[1] < 0 else 6

        velocity = traci.vehicle.getSpeed(self.av_id)
        acceleration = traci.vehicle.getAcceleration(self.av_id)
        road_id = traci.vehicle.getLaneIndex(self.av_id)
        distance = traci.vehicle.getDistance(self.av_id)

        return action_rule, velocity, acceleration, road_id, distance