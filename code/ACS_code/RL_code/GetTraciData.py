import sys
import os
import traci
import math


class GetTraciData(object):
    def __init__(self):
        self.Check()
        self.sumocfgfile = r"."
        self.render_cmd = "sumo"
        self.step_length = "0.1"
        self.av_id = 'DRL_AV'
        self.lateral_resolution = "36"
        self.lane_width = [3.6, 3.6, 3.6]
        self.maxspeed = 120
        self.preheat_time = 0
        self.minspeed = 80  # km/h
        self.state_last = []

    def StartSimulation(self, If_update=False, seed=1):
        """
        Start simulation
        :return: null
        """
        if If_update:
            traci.start([self.render_cmd, "--step-length", self.step_length, "--lateral-resolution", "6", "-c",
                         self.sumocfgfile, "--seed", str(seed)])
        else:
            traci.start(
                [self.render_cmd, "--step-length", self.step_length, "--lateral-resolution", "6", "-c",
                 self.sumocfgfile, "--random"])

    def StepSimulation(self, action):
        self.ControlVehicle(action)
        traci.simulationStep()

    def GetState(self):
        left_front_info = traci.vehicle.getNeighbors(self.av_id, 0b00000010)  # 此处的意思的二进制
        left_rear_info = traci.vehicle.getNeighbors(self.av_id, 0b00000000)
        right_front_info = traci.vehicle.getNeighbors(self.av_id, 0b00000011)
        right_rear_info = traci.vehicle.getNeighbors(self.av_id, 0b00000001)

        leader_info = (traci.vehicle.getLeader(self.av_id, 0),)

        info = [leader_info if leader_info[0] is not None else (), left_front_info, left_rear_info, right_front_info,
                right_rear_info]
        state = self.GetSurround(info)
        self.state_last = state

        return state

    def ControlVehicle(self, action):
        # 0 1 2
        LaneId = traci.vehicle.getLaneIndex(self.av_id)

        LaneID_Target = LaneId + action[0]
        LaneID_Target = LaneID_Target if (0 <= LaneID_Target <= 2) else (0 if LaneID_Target < 0 else 2)

        LaneChange_Distance = (LaneID_Target - LaneId) * (self.lane_width[LaneId] + self.lane_width[LaneID_Target]) \
                              / 2 - traci.vehicle.getLateralLanePosition(self.av_id)

        traci.vehicle.changeSublane(self.av_id, LaneChange_Distance)

        action_speed = action[1]

        speed_av = traci.vehicle.getSpeed(self.av_id)
        # min
        if speed_av + float(self.step_length) * action[1] < self.minspeed / 3.6:
            action_speed = (- speed_av + self.minspeed / 3.6) / float(self.step_length)

        # max
        if speed_av + float(self.step_length) * action[1] > self.maxspeed / 3.6:
            action_speed = (self.maxspeed / 3.6 - speed_av) / float(self.step_length)

        traci.vehicle.setAcceleration(self.av_id, action_speed, 0.1)

    def GetSurround(self, info):
        # getLanePosition
        accelerate_av = traci.vehicle.getAcceleration(self.av_id)
        velocity_av = traci.vehicle.getSpeed(self.av_id)
        angel_av = traci.vehicle.getAngle(self.av_id) - 90

        state = [velocity_av, accelerate_av, angel_av]

        LaneId = traci.vehicle.getLaneIndex(self.av_id)
        y_av = traci.vehicle.getLateralLanePosition(self.av_id)
        width_av = traci.vehicle.getWidth(self.av_id)
        position_av = traci.vehicle.getPosition(self.av_id)

        if LaneId == 0:
            width_left = self.lane_width[0] / 2 - y_av + self.lane_width[1] + self.lane_width[2] - width_av / 2
            width_right = -self.lane_width[0] / 2 - y_av + width_av / 2
        elif LaneId == 2:
            width_left = self.lane_width[2] / 2 - y_av - width_av / 2
            width_right = -self.lane_width[2] / 2 - y_av + width_av / 2 - self.lane_width[1] - self.lane_width[0]
        else:
            width_left = self.lane_width[1] / 2 - y_av - width_av / 2 + self.lane_width[2]
            width_right = -self.lane_width[1] / 2 - y_av + width_av / 2 - self.lane_width[0]

        num = 0

        for i in range(0, len(info)):
            if len(info[i]) == 0:
                if num == 0 or num == 1 or num == 3:
                    state.append(100)
                    state.append(0 if i == 0 else (width_left if i // 3 == 0 else width_right))

                    state.append(traci.vehicle.getSpeed(self.av_id) - self.maxspeed / 3.6)  # 前车速度减后车速度
                else:
                    state.append(-100)
                    state.append(0 if i == 0 else (width_left if i // 3 == 0 else width_right))

                    state.append(traci.vehicle.getSpeed(self.av_id))
            else:
                position = traci.vehicle.getPosition(info[i][0][0])
                length = traci.vehicle.getLength(info[i][0][0])
                width = traci.vehicle.getWidth(info[i][0][0])
                distance_x = position[0] - position_av[0] - length * (position[0] - position_av[0]) / abs(
                    position[0] - position_av[0] + 0.000001)
                distance_y = position[1] - position_av[1] - 0.5 * (width_av + width) * (
                        position[1] - position_av[1]) / abs(position[1] - position_av[1] + 0.000001)

                # <100m
                if abs(distance_x) <= 100:
                    state.append(distance_x)
                    state.append(distance_y)

                    state.append(self.GetVelocity(info[i][0][0]))
                else:
                    state.append(100 * distance_x / abs(distance_x + 0.0000001))
                    state.append(0 if i == 0 else (width_left if i // 3 == 0 else width_right))

                    if distance_x / abs(distance_x + 0.0000001) < 0:
                        state.append(traci.vehicle.getSpeed(self.av_id))
                    else:
                        state.append(traci.vehicle.getSpeed(self.av_id) - self.maxspeed / 3.6)

        return state

    def GetVelocity(self, sv_id):
        velocity = traci.vehicle.getSpeed(sv_id)
        velocity_av = traci.vehicle.getSpeed(self.av_id)

        return velocity_av - velocity

    def CloseSimulation(self):
        traci.close()

    def Preheat(self, time_length=0):
        traci.simulationStep(time_length)
        self.preheat_time = time_length  # update

        while False if self.av_id in traci.vehicle.getIDList() else True:
            traci.simulationStep()
            self.preheat_time += float(self.step_length)

        traci.vehicle.setLaneChangeMode(self.av_id, 0b000000000000)  # close lane change model
        traci.vehicle.setSpeedMode(self.av_id, 0b00000)

        self.maxspeed = traci.vehicle.getMaxSpeed(self.av_id) * 3.6
        if self.render_cmd == "sumo-gui": traci.gui.trackVehicle("View #0", self.av_id)

    def CollisionDetection(self):
        Collision_List = traci.simulation.getCollidingVehiclesIDList()
        done_collision = self.av_id in Collision_List
        return done_collision

    def GetTimeDone(self, simulation_time):
        done_step = False
        time_length = traci.simulation.getTime()

        if time_length >= self.preheat_time + simulation_time:
            done_step = True

        return done_step

    def RuleModel(self):
        action_rule = [1, 0]
        # IDM
        a_max, d_min, T, b, v = 3, 2, 1.1, 3, self.state_last[0]
        delta_v = self.state_last[5]  # x, y, v

        distance_expect = d_min + T * v + v * delta_v / (2 * math.sqrt(a_max * b))

        action_rule[1] = a_max * (1 - (v * 3.6 / self.maxspeed) ** 4 - (distance_expect / self.state_last[3]) ** 2)

        action_rule[1] = action_rule[1] + 3 if -3 < action_rule[1] < 3 else 0 if action_rule[1] < 0 else 6

        velocity = traci.vehicle.getSpeed(self.av_id)
        acceleration = traci.vehicle.getAcceleration(self.av_id)

        return action_rule, velocity, acceleration

    @staticmethod
    def Check():
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")


if __name__ == '__main__':
    gtd = GetTraciData()
    gtd.StartSimulation()
    gtd.Preheat(57)
    gtd.GetSurround()
    gtd.CloseSimulation()
