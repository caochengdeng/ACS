class ActionProcessor:

    def process_action(self, action):
        try:
            lane_change = action[0] - 1
            acceleration = action[1] - 3
            return [lane_change, acceleration]
        except Exception:
            return [None, action]