import traci


class StateCalculator:
    """
    Class to calculate and return the current environment state.
    """

    def __init__(self):
        self.av_id = 'DRL_AV'
        self.lane_width = [3.6, 3.6, 3.6]

    def get_state(self, maxspeed):
        """
        Get current vehicle and surrounding vehicles' states.
        """
        av_id = self.av_id
        left_front_info = traci.vehicle.getNeighbors(av_id, 0b00000010)
        left_rear_info = traci.vehicle.getNeighbors(av_id, 0b00000000)
        right_front_info = traci.vehicle.getNeighbors(av_id, 0b00000011)
        right_rear_info = traci.vehicle.getNeighbors(av_id, 0b00000001)

        leader_info = (traci.vehicle.getLeader(av_id, 0),)
        info = [
            leader_info if leader_info[0] is not None else (),
            left_front_info, left_rear_info,
            right_front_info, right_rear_info
        ]

        return self._calculate_surrounding(info, maxspeed)

    def _calculate_surrounding(self, info, maxspeed):
        """
        Calculate relative distances and velocities of surrounding vehicles.
        """
        av_id = self.av_id
        position_av = traci.vehicle.getPosition(av_id)
        width_av = traci.vehicle.getWidth(av_id)
        velocity_av = traci.vehicle.getSpeed(av_id)
        accelerate_av = traci.vehicle.getAcceleration(av_id)
        angel_av = traci.vehicle.getAngle(av_id) - 90

        LaneId = traci.vehicle.getLaneIndex(av_id)
        y_av = traci.vehicle.getLateralLanePosition(av_id)

        width_left = 0
        width_right = 0
        if LaneId == 0:
            width_left = self.lane_width[0] / 2 - y_av + self.lane_width[1] + self.lane_width[2] - width_av / 2
            width_right = -self.lane_width[0] / 2 - y_av + width_av / 2
        elif LaneId == 2:
            width_left = self.lane_width[2] / 2 - y_av - width_av / 2
            width_right = -self.lane_width[2] / 2 - y_av + width_av / 2 - self.lane_width[1] - self.lane_width[0]
        else:
            width_left = self.lane_width[1] / 2 - y_av - width_av / 2 + self.lane_width[2]
            width_right = -self.lane_width[1] / 2 - y_av + width_av / 2 - self.lane_width[0]

        state = [velocity_av, accelerate_av, angel_av]

        for i in range(len(info)):
            if len(info[i]) == 0:
                state.append(100)
                state.append(0 if i == 0 else width_left if i // 3 == 0 else width_right)
                state.append(maxspeed / 3.6)
            else:
                sv_id = info[i][0][0]
                position = traci.vehicle.getPosition(sv_id)
                length = traci.vehicle.getLength(sv_id)
                width = traci.vehicle.getWidth(sv_id)
                position_av = traci.vehicle.getPosition(av_id)

                dx = position[0] - position_av[0] - length * (position[0] - position_av[0]) / abs(
                    position[0] - position_av[0] + 0.000001
                )
                dy = position[1] - position_av[1] - 0.5 * (width_av + width) * (
                    position[1] - position_av[1]
                ) / abs(position[1] - position_av[1] + 0.000001)

                if abs(dx) <= 100:
                    state.extend([dx, dy, self.get_velocity_diff(sv_id)])
                else:
                    state.extend([
                        100 * dx / abs(dx),
                        0 if i == 0 else width_left if i // 3 == 0 else width_right,
                        velocity_av
                    ])

        return state