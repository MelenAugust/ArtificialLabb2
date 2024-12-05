from cmath import phase

import poker_environment as pe_
from poker_environment import AGENT_ACTIONS, BETTING_ACTIONS
import copy
from collections import deque



"""
Player class
"""
class PokerPlayer(object):
    def __init__(self, current_hand_=None, stack_=300, action_=None, action_value_=None):
        self.current_hand = current_hand_
        self.current_hand_type = []
        self.current_hand_strength = []
        self.stack = stack_
        self.action = action_
        self.action_value = action_value_

    """
    identify agent hand and evaluate it's strength
    """
    def evaluate_hand(self):
        self.current_hand_type = pe_.identify_hand(self.current_hand)
        self.current_hand_strength = pe_.Types[self.current_hand_type[0]]*len(pe_.Ranks) + pe_.Ranks[self.current_hand_type[1]]

    """
    return possible actions, fold if there is not enough money...
    """
    def get_actions(self):
        actions_ = []
        for _action_ in AGENT_ACTIONS:
            if _action_[:3] == 'BET' and int(_action_[3:])>=(self.stack):
                actions_.append('FOLD')
            else:
                actions_.append(_action_)
        return set(actions_)

"""
Game State class
"""
class GameState(object):
    def __init__(self,
                 nn_current_hand_=None,
                 nn_current_bidding_=None,
                 phase_ = None,
                 pot_=None,
                 acting_agent_=None,
                 parent_state_=None,
                 children_state_=None,
                 agent_=None,
                 opponent_=None):

        self.nn_current_hand = nn_current_hand_
        self.nn_current_bidding = nn_current_bidding_
        self.phase = phase_
        self.pot = pot_
        self.acting_agent = acting_agent_
        self.parent_state = parent_state_
        self.children = children_state_
        self.agent = agent_
        self.opponent = opponent_
        self.showdown_info = None

    """
    draw 10 cards randomly from a deck
    """
    def dealing_cards(self):

        if self.nn_current_hand < 20:
            #print("random hand  ", self.nn_current_hand)
            # randomly generated hands
            agent_hand, opponent_hand = pe_.generate_2hands()
        else:
            # fixed_hands or use the function below
            print("fixed hand ", self.nn_current_hand)
            agent_hand, opponent_hand = pe_.fixed_hands[self.nn_current_hand]
        self.agent.current_hand = agent_hand
        self.agent.evaluate_hand()
        self.opponent.current_hand = opponent_hand
        self.opponent.evaluate_hand()

    """
    deals 5 cards to the agent and opponent an evaluates hand type and strength
    """

    def dealing_cards_fixed(self):
        if self.nn_current_hand >= len(pe_.fixed_hands):
            raise IndexError(f"nn_current_hand ({self.nn_current_hand}) out of bounds for fixed_hands")
        self.agent.current_hand = pe_.fixed_hands[self.nn_current_hand][0]
        self.agent.evaluate_hand()
        self.opponent.current_hand = pe_.fixed_hands[self.nn_current_hand][1]
        self.opponent.evaluate_hand()

    """
    SHOWDOWN phase, assign pot to players
    """
    def showdown(self):

        if self.agent.current_hand_strength == self.opponent.current_hand_strength:
            self.showdown_info = 'draw'
            if self.acting_agent == 'agent':
                self.agent.stack += (self.pot - 5) / 2. + 5
                self.opponent.stack += (self.pot - 5) / 2.
            else:
                self.agent.stack += (self.pot - 5) / 2.
                self.opponent.stack += (self.pot - 5) / 2. + 5
        elif self.agent.current_hand_strength > self.opponent.current_hand_strength:
            self.showdown_info = 'agent win'
            self.agent.stack += self.pot
        else:
            self.showdown_info = 'opponent win'
            self.opponent.stack += self.pot

    # print out necessary information of this game state
    def print_state_info(self):

        if self.phase == "INIT_DEALING" or self.nn_current_hand is None:
            print(f"\n--- Hand {self.nn_current_hand} ---\n")
            return
        print(f"Acting agent: {self.acting_agent}")
        print(f"Agent move: {self.agent.action}")
        print(f"Opponent move: {self.opponent.action}")
        print(f"Phase: {self.phase}")
        print(f"Pot: {self.pot}")
        print(f"Agent stack: {self.agent.stack}, Opponent stack: {self.opponent.stack}")

        if self.phase == 'SHOWDOWN':
            print(f"--- Showdown ---")
            print(f"Agent hand: {self.agent.current_hand_type} ({self.agent.current_hand})")
            print(f"Agent hand strength: {self.agent.current_hand_strength}")
            print(f"Opponent hand: {self.opponent.current_hand_type} ({self.opponent.current_hand})")
            print(f"Opponent hand strength: {self.opponent.current_hand_strength}")
            print(f"Showdown result: {self.showdown_info}")
        else:
            print(f"Agent last action: {self.agent.action} ({self.agent.action_value})")
            print(f"Opponent last action: {self.opponent.action} ({self.opponent.action_value})")

        print(f"\n--- Hand {self.nn_current_hand} ---\n")


# copy given state in the argument
def copy_state(game_state):
    _state = copy.copy(game_state)
    _state.agent = copy.copy(game_state.agent)
    _state.opponent = copy.copy(game_state.opponent)
    return _state

"""
successor function for generating next state(s)
"""
def get_next_states(last_state):
    if last_state.nn_current_hand > 4:
        return []

    if last_state.phase == 'SHOWDOWN' or last_state.acting_agent == 'opponent' or last_state.phase == 'INIT_DEALING':

        # NEW BETTING ROUND, AGENT ACT FIRST

        states = []

        for _action_ in last_state.agent.get_actions():


            _state_ = copy_state(last_state)
            _state_.acting_agent = 'agent'

            if last_state.phase == 'SHOWDOWN' or last_state.phase == 'INIT_DEALING':
                _state_.dealing_cards()
                #_state_.dealing_cards_fixed()

            if _action_ == 'CALL':

                _state_.phase = 'SHOWDOWN'
                _state_.agent.action = _action_
                _state_.agent.action_value = 5
                _state_.agent.stack -= 5
                _state_.pot += 5

                _state_.showdown()

                _state_.nn_current_hand += 1
                _state_.nn_current_bidding = 0
                _state_.pot = 0
                _state_.parent_state = last_state
                states.append(_state_)

            elif _action_ == 'FOLD':

                _state_.phase = 'SHOWDOWN'
                _state_.agent.action = _action_
                _state_.opponent.stack += _state_.pot

                _state_.nn_current_hand += 1
                _state_.nn_current_bidding = 0
                _state_.pot = 0
                _state_.parent_state = last_state
                states.append(_state_)


            elif _action_ in BETTING_ACTIONS:

                _state_.phase = 'BIDDING'
                _state_.agent.action = _action_
                _state_.agent.action_value = int(_action_[3:])
                _state_.agent.stack -= int(_action_[3:])
                _state_.pot += int(_action_[3:])

                _state_.nn_current_bidding += 1
                _state_.parent_state = last_state
                states.append(_state_)

            else:

                print('...unknown action...')
                exit()

        return states

    elif last_state.phase == 'BIDDING' and last_state.acting_agent == 'agent':

        states = []
        _state_ = copy_state(last_state)
        _state_.acting_agent = 'opponent'

        opponent_action, opponent_action_value = pe_.poker_strategy_example(last_state.opponent.current_hand_type[0],
                                                                            last_state.opponent.current_hand_type[1],
                                                                            last_state.opponent.stack,
                                                                            last_state.agent.action,
                                                                            last_state.agent.action_value,
                                                                            last_state.agent.stack,
                                                                            last_state.pot,
                                                                            last_state.nn_current_bidding)

        if opponent_action =='CALL':

            _state_.phase = 'SHOWDOWN'
            _state_.opponent.action = opponent_action
            _state_.opponent.action_value = 5
            _state_.opponent.stack -= 5
            _state_.pot += 5

            _state_.showdown()

            _state_.nn_current_hand += 1
            _state_.nn_current_bidding = 0
            _state_.pot = 0
            _state_.parent_state = last_state
            states.append(_state_)

        elif opponent_action == 'FOLD':

            _state_.phase = 'SHOWDOWN'

            _state_.opponent.action = opponent_action
            _state_.agent.stack += _state_.pot

            _state_.nn_current_hand += 1
            _state_.nn_current_bidding = 0
            _state_.pot = 0
            _state_.parent_state = last_state
            states.append(_state_)

        elif opponent_action + str(opponent_action_value) in BETTING_ACTIONS:

            _state_.phase = 'BIDDING'

            _state_.opponent.action = opponent_action
            _state_.opponent.action_value = opponent_action_value
            _state_.opponent.stack -= opponent_action_value
            _state_.pot += opponent_action_value

            _state_.nn_current_bidding += 1
            _state_.parent_state = last_state
            states.append(_state_)

        else:
            print('unknown_action')
            exit()
        return states

def bfs_search_with_history(initial_state):
    queue = deque([(initial_state, [], 0)])  # (state, path, node_id)
    visited = set()
    nodes_expanded = 0
    node_history = {}

    node_id = 0
    node_history[node_id] = (None, initial_state)  # Root node

    while queue:

        current_state, path, current_node_id = queue.popleft()
        nodes_expanded += 1

        if is_goal_state(current_state):
            print(f"BFS found a solution! Nodes expanded: {nodes_expanded}")
            return path + [(current_node_id, current_state)], nodes_expanded, node_history


        visited.add(id(current_state))


        next_states = get_next_states(current_state)

        for state in next_states:
            if id(state) not in visited:

                node_id += 1
                node_history[node_id] = (current_node_id, state)

                # Add the new state to the queue
                queue.append((state, path + [(current_node_id, current_state)], node_id))

    print(f"No solution found. Nodes expanded: {nodes_expanded}")
    return None, nodes_expanded, node_history

def dfs_search_with_history(initial_state):
    stack = [(initial_state, [], 0)]  # (state, path, node_id)
    visited = set()
    nodes_expanded = 0
    node_history = {}

    node_id = 0
    node_history[node_id] = (None, initial_state)  # Root node

    while stack:

        current_state, path, current_node_id = stack.pop()
        nodes_expanded += 1


        if is_goal_state(current_state):
            print(f"DFS found a solution! Nodes expanded: {nodes_expanded}")
            return path + [(current_node_id, current_state)], nodes_expanded, node_history

        visited.add(id(current_state))

        next_states = get_next_states(current_state)

        for state in next_states:
            if id(state) not in visited:

                node_id += 1
                node_history[node_id] = (current_node_id, state)


                stack.append((state, path + [(current_node_id, current_state)], node_id))

    print(f"No solution found. Nodes expanded: {nodes_expanded}")
    return None, nodes_expanded, node_history


def count_all_states_up_to_hand_4(initial_state):

    count = 0
    queue = deque([initial_state])  # BFS queue
    visited = set()
    total_states = 0  # Counter for all states generated

    while queue:
        current_state = queue.popleft()




        # Skip already visited states
        if current_state in visited:
            continue
        visited.add(current_state)


        if current_state.nn_current_hand == 5:

            continue
        total_states += 1


        next_states = get_next_states(current_state)
        queue.extend(next_states)

    return total_states,count











def is_goal_state(state):

    return  state.agent.stack >= 200 and state.nn_current_hand <= 4

"""
Game flow:
Two agents will keep playing until one of them lose 100 coins or more.
"""

MAX_HANDS = 4
INIT_AGENT_STACK = 100

# initialize 2 agents and a game_state
agent = PokerPlayer(current_hand_=None, stack_=INIT_AGENT_STACK, action_=None, action_value_=None)
opponent = PokerPlayer(current_hand_=None, stack_=INIT_AGENT_STACK, action_=None, action_value_=None)


init_state = GameState(nn_current_hand_=1,
                       nn_current_bidding_=0,
                       phase_ = 'INIT_DEALING',
                       pot_=0,
                       acting_agent_=None,
                       agent_=agent,
                       opponent_=opponent,
                       )





print("Starting BFS search...")
solution_path, nodes_expanded, node_history = dfs_search_with_history(init_state)

if solution_path:
 print(f"Solution path found! Nodes expanded: {nodes_expanded}")
 for idx, (node_id, state) in enumerate(solution_path):
     print(f"Step {idx + 1}: Node {node_id}")
     state.print_state_info()
 else:
     print("No solution found.")

#total_states,count = count_all_states_up_to_hand_4(init_state)

# Print the result
#print(f"Total states generated up to hand 4: {total_states}")







"""
Perform searches
"""






