from copy import deepcopy
from itertools import product
import numpy as np
import random


class DFS:
    def __init__(self, board, figures_list):
        self.board = board
        self.figures_list = figures_list

    def check_constraints(self, target_coordinates):
        for idx, figure in enumerate(self.figures_list):
            if figure.attack(target_coordinates, self.board.shape[0]):
                return True
        return False

    def dfs(self, figure_idx):

        if figure_idx >= len(self.figures_list):
            return self.board

        for i, j in product(range(self.board.shape[0]), range(self.board.shape[0])):
            raw_coords = tuple(zip(*np.where(self.board == '0.0')))
            non_raw_coords = tuple(zip(*np.where(self.board != '0.0')))

            tmp_fig = self.figures_list[figure_idx]
            tmp_fig._coordinates = (i, j)

            if ((i, j) in raw_coords and not len(set(non_raw_coords) &
                                                 set(tmp_fig.get_step_coordinates(self.board.shape[0]))
                                                 )):
                is_possible_attack = self.check_constraints((i, j))

                if not is_possible_attack:
                    self.board[i, j] = self.figures_list[figure_idx]._name
                    self.figures_list[figure_idx]._coordinates = (i, j)

                    coords = self.dfs(figure_idx + 1)
                    if coords is not None:
                        return coords

        return None

    def main(self):

        coords_res = self.dfs(0)

        coords_res = [[' ' if el == '0.0' else el for el in el1] for el1 in coords_res]

        board_gen = ('/').join([('').join([el if el != '0.0' else ' ' for el in row]) for row in coords_res])

        return board_gen


class Agent:
    def __init__(self, figure, priority, neighbor_list, domain, board):
        self.figure = figure
        self.priority = priority
        self.neighbors_list = deepcopy(neighbor_list)
        self.agent_domain = domain
        self.consistent = True
        self.constraints = []
        self.board = board

    def check_consistency(self):
        self.consistent = True
        for neighbor in self.neighbors_list:
            if (neighbor.figure.attack(self.figure._coordinates, self.board.shape[0])
                    or self.figure._coordinates == neighbor.figure._coordinates):
                self.consistent = False

    def change_value(self):
        self.figure._coordinates = random.choice(self.agent_domain)

    def check_local_view(self):
        self.check_consistency()
        if not self.consistent:
            old_value = self.figure._coordinates
            for domain in self.agent_domain:
                self.figure._coordinates = domain
                self.check_consistency()
                if self.consistent:
                    break
            if not self.consistent:
                for agents in self.neighbors_list:
                    if agents.priority == self.priority - 1:
                        master_agent = agents
                backtrack(master_agent)
            else:
                for agents in self.neighbors_list:
                    send_handle_ok(agents)

    def handle_ok(self, agent):
        for neighbor in self.neighbors_list:
            if neighbor.figure._name == agent.figure._name:
                neighbor.figure._coordinates = agent.figure._coordinates
                neighbor.priority = agent.priority
                self.check_local_view()

    def handle_nogood(self, nogood_list):
        self.constraints.append(nogood_list)
        for agent_in_nogood in nogood_list:
            if any(neighbor_agent['name'] == agent_in_nogood[0] for neighbor_agent in self.neighbors_list):
                pass
            else:
                self.neighbors_list.append(
                    {'name': agent_in_nogood[0], 'value': agent_in_nogood[1], 'priority': agent_in_nogood[2]})
        old_value = self.figure._coordinates
        self.check_local_view()
        if old_value != self.figure._coordinates:
            for agents in self.neighbors_list:
                send_handle_ok(agents)


def backtrack(master_agent):
    higher_priority_agents = []
    higher_domains = []
    for neighbor in master_agent.neighbors_list:
        if neighbor.priority < master_agent.priority:
            higher_priority_agents.append(neighbor)
    for agent in higher_priority_agents:
        higher_domains.append(agent.figure._coordinates)
    if master_agent.agent_domain == higher_domains:
        print("no_solution")
    else:
        if not (master_agent.figure._coordinates in higher_domains):
            print("no_solution")
            return 0
        while master_agent.figure._coordinates in higher_domains:
            master_agent.figure._coordinates = random.choice(master_agent.agent_domain)
        for agents in master_agent.neighbors_list:
            if agents.priority > master_agent.priority:
                send_handle_ok(master_agent)


def send_handle_ok(agent):
    for agents in agent.neighbors_list:
        if agents:
            if agents.priority > agent.priority:
                agents.handle_ok(agent)


class ABT:

    def __init__(self, board, figures_mapping):
        self.board = board
        self.figures_mapping = figures_mapping

    def abt(self, agent_list):
        all_domain = list(product(range(self.board.shape[0]), range(self.board.shape[0])))

        processed_agents = []
        for i, agent in enumerate(agent_list):
            agent.check_consistency()
            send_handle_ok(agent)
            processed_agents.append(agent)

            new_domain = list(set(all_domain) - set([ag.figure._coordinates for ag in processed_agents]))
            agent.agent_domain = new_domain

            new_neighbours = []
            for neighbour in agent.neighbors_list:
                neighbour.agent_domain = new_domain

                new_neighbours.append(neighbour)

            agent.neighbors_list = new_neighbours

        return agent_list

    def prepare_agents(self):
        all_domain = list(product(range(self.board.shape[0]), range(self.board.shape[0])))
        figures_mapping_last_changed_names = deepcopy(self.figures_mapping)

        for coords, figure in figures_mapping_last_changed_names.items():
            new_fig = deepcopy(figure)
            new_fig._name = '{}_{}'.format(figure._name, coords)
            figures_mapping_last_changed_names[coords] = new_fig

        agents = [Agent(figure, i + 1, [], all_domain, self.board)
                  for i, figure in enumerate(figures_mapping_last_changed_names.values())
                  ]

        for idx, agent in enumerate(agents):
            agent.neighbors_list.extend([ag for ag_idx, ag in enumerate(agents) if ag_idx != idx])

        return agents

    def main(self):
        agents = self.prepare_agents()

        agents = self.abt(agents)

        coords_sol = np.zeros((self.board.shape[0], self.board.shape[0])).astype(str)
        coords_sol = [[' ' if el1 == '0.0' else el1 for el1 in el] for el in coords_sol]

        for agent in agents:
            coords_sol[agent.figure._coordinates[0]][agent.figure._coordinates[1]] = agent.figure._name.split('_')[0]

        sol_gen = ('/').join([('').join([el if el != '0.0' else ' ' for el in row]) for row in coords_sol])

        return sol_gen
