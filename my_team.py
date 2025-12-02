# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


def create_team(first_index, second_index, is_red,
                first='OffenseAgent', second='DefenseAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


class ReflexAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        if Directions.STOP in best_actions and len(best_actions) > 1:
            best_actions.remove(Directions.STOP)

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        return util.Counter()

    def get_weights(self, game_state, action):
        return {}


class OffenseAgent(ReflexAgent):
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.stuck_counter = 0
        self.last_pos = None
        self.history = []

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Exploration History
        if my_pos in self.history:
            features['visited_penalty'] = 1
        if my_pos == self.start:
            self.history = []

        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        # Ghost awareness
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = []
        for a in enemies:
            if not a.is_pacman and a.get_position() is not None:
                # If timer is low (< 5), treat as dangerous even if scared
                if a.scared_timer <= 5:
                    ghosts.append(a)

        dist_to_ghost = float('inf')
        closest_ghost_pos = None
        for ghost in ghosts:
            d = self.get_maze_distance(my_pos, ghost.get_position())
            if d < dist_to_ghost:
                dist_to_ghost = d
                closest_ghost_pos = ghost.get_position()

        # Immediate danger
        if dist_to_ghost <= 5:
            features['ghost_distance'] = 2.0 / (dist_to_ghost + 0.1)

        # Safe food filtering
        target_food = []
        for food in food_list:
            if closest_ghost_pos:
                my_dist = self.get_maze_distance(my_pos, food)
                ghost_dist = self.get_maze_distance(closest_ghost_pos, food)
                if my_dist < ghost_dist:
                    target_food.append(food)
            else:
                target_food.append(food)

        if not target_food:
            target_food = food_list

        if len(target_food) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in target_food])
            features['distance_to_food'] = min_distance

        # Home
        layout_width = game_state.data.layout.width
        mid_x = layout_width // 2
        home_x = mid_x - 1 if self.red else mid_x + 1
        home_locations = [(home_x, y) for y in range(game_state.data.layout.height)
                          if not game_state.has_wall(home_x, y)]

        if home_locations:
            features['distance_to_home'] = min([self.get_maze_distance(my_pos, loc) for loc in home_locations])

        # Capsules
        capsules = self.get_capsules(successor)
        if len(capsules) > 0:
            features['distance_to_capsule'] = min([self.get_maze_distance(my_pos, cap) for cap in capsules])

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        weights = {
            'successor_score': 100,
            'distance_to_food': -1,
            'ghost_distance': -1000,
            'stop': -100,
            'reverse': -2,
            'distance_to_home': 0,
            'distance_to_capsule': -2,
            'visited_penalty': -5
        }

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_left = len(self.get_food(game_state).as_list())
        time_left = game_state.data.timeleft

        if self.last_pos == my_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self.last_pos = my_pos

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        active_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer <= 5]
        danger_close = False
        for ghost in active_ghosts:
            if self.get_maze_distance(my_pos, ghost.get_position()) < 5:
                danger_close = True

        should_return = False
        if my_state.num_carrying > 0:
            if my_state.num_carrying >= 5:
                should_return = True
            elif food_left <= 2:
                should_return = True
            elif time_left < 200:
                should_return = True
            elif danger_close:
                should_return = True
            elif self.stuck_counter > 5:
                should_return = True

        if should_return:
            weights['distance_to_home'] = -50
            weights['distance_to_food'] = 0
            weights['successor_score'] = 0
            weights['visited_penalty'] = 0

        return weights

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        self.history.append(my_pos)
        if len(self.history) > 20:
            self.history.pop(0)
        return super().choose_action(game_state)


class DefenseAgent(ReflexAgent):
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.target_enemy_pos = None
        self.last_food_list = self.get_food_you_are_defending(game_state).as_list()

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Invisible Enemy Tracking
        current_food_list = self.get_food_you_are_defending(game_state).as_list()
        if len(current_food_list) < len(self.last_food_list):
            missing = set(self.last_food_list) - set(current_food_list)
            if missing: self.target_enemy_pos = list(missing)[0]

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        visible = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(visible)

        if len(visible) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in visible]
            features['invader_distance'] = min(dists)
            self.target_enemy_pos = visible[dists.index(min(dists))].get_position()
        elif self.target_enemy_pos:
            if my_pos == self.target_enemy_pos:
                self.target_enemy_pos = None
            else:
                features['distance_to_last_known'] = self.get_maze_distance(my_pos, self.target_enemy_pos)
        else:
            # Calculate the border X
            mid_x = game_state.data.layout.width // 2
            patrol_x = mid_x - 1 if self.red else mid_x + 1

            # Calculate the "Weighted Y" based on food clusters
            defending_food = self.get_food_you_are_defending(game_state).as_list()
            if len(defending_food) > 0:
                avg_y = sum([f[1] for f in defending_food]) / len(defending_food)
                patrol_y = int(avg_y)
            else:
                patrol_y = game_state.data.layout.height // 2

            # Ensure patrol point is not a wall.
            # If (patrol_x, patrol_y) is a wall, look for nearest valid point vertically
            patrol_pos = (patrol_x, patrol_y)
            if game_state.has_wall(int(patrol_x), int(patrol_y)):
                # Search locally for an open spot
                neighbors = []
                for i in range(1, 6):
                    if not game_state.has_wall(int(patrol_x), int(patrol_y + i)): neighbors.append(
                        (patrol_x, patrol_y + i))
                    if not game_state.has_wall(int(patrol_x), int(patrol_y - i)): neighbors.append(
                        (patrol_x, patrol_y - i))

                if neighbors:
                    patrol_pos = min(neighbors, key=lambda p: self.get_maze_distance(my_pos, p))
                else:
                    patrol_pos = self.start

            features['distance_to_patrol'] = self.get_maze_distance(my_pos, patrol_pos)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        self.last_food_list = self.get_food_you_are_defending(game_state).as_list()
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'distance_to_last_known': -5,
            'distance_to_patrol': -2,
            'stop': -100,
            'reverse': -2
        }