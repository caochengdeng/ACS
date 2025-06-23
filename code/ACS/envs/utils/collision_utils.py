import traci


class CollisionDetector:
    """
    Class to detect collisions and determine simulation termination.
    """

    def check_collision(self):
        """
        Detect if a collision occurred involving the AV.
        """
        Collision_List = traci.simulation.getCollidingVehiclesIDList()
        done_collision = self.av_id in Collision_List
        for v_id in Collision_List:
            if v_id != self.av_id:
                traci.vehicle.remove(v_id)
        return done_collision

    def check_time_done(self, current_time, preheat_time, simulation_time):
        """
        Check if simulation has reached its time limit.
        """
        return current_time >= (preheat_time + simulation_time)