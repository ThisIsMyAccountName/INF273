from time import time
from copy import deepcopy
from random import sample, randint, shuffle, random
from collections import defaultdict
from pdp_utils.pdp_utils.Utils import feasibility_check, cost_function
from init import init_problem, init_solution, pprint
import numpy as np
from helpers import *

def pick_random_cargo(list_sol, prob, memo):
	picks = [i for i in range(len(list_sol)) if len(list_sol[i]) > 2]
	vessel = sample(picks, 1)[0]
	cargo = sample(list_sol[vessel], 1)[0]
	memo.add((cargo, vessel))
	return cargo, vessel

def find_cargo_to_move(list_solution, prob):
	if list_solution[-1]:
		if random() < 0.9:
			return sample(list_solution[-1], 1)[0], len(list_solution) - 1
		Cargo = prob['Cargo']
		most_expensive = max(list(zip(Cargo[:, 3], list_solution[-1])), key=lambda x: x[0])
		return most_expensive[1], len(list_solution) - 1
	return find_most_expensive_cargo_to_move(list_solution, prob)
	

def optimize_solution(org_solution, prob, cargo_dict, iterations=200, print_iter=False):
	list_solution = org_solution.copy()
	best_cost = cost_function(list_to_solution(list_solution), prob)
	for iter in range(iterations):
		if iter % (iterations // 10) == 0 and print_iter:
			print(f"Iteration: {iter}, cost {cost_function(list_to_solution(list_solution), prob)}")
		solution = deepcopy(list_solution)
		cargo_to_move, move_from = find_cargo_to_move(list_solution, prob)
		move_to = sample(cargo_dict[cargo_to_move], 1)[0]
		remove_elem_from(solution, cargo_to_move, move_from)
		feasebility, new_sol = insert_number_at_all_positions_pick_first(solution, move_to, move_from, cargo_to_move, prob)
		if feasebility and (new_cost:= cost_function(list_to_solution(new_sol), prob)) < best_cost:
			list_solution = new_sol
			best_cost = new_cost

	return list_solution


def run_tests(num_tests, iterations, prob, print_iter=False):

	n_nodes = prob['n_nodes']
	n_vehicles = prob['n_vehicles']
	n_calls = prob['n_calls']
	Cargo =	prob['Cargo']
	TravelTime = prob['TravelTime']
	FirstTravelTime = prob['FirstTravelTime']
	VesselCapacity = prob['VesselCapacity']
	LoadingTime = prob['LoadingTime']
	UnloadingTime = prob['UnloadingTime']
	VesselCargo = prob['VesselCargo']
	TravelCost = prob['TravelCost']
	FirstTravelCost = prob['FirstTravelCost']
	PortCost = prob['PortCost']
	avg_times, solutions = [], []
	best_solution = init_solution(n_vehicles, n_calls)
	cargo_dict = cargo_to_vessel(VesselCargo, n_calls)
	for test in range(num_tests):
		start_time = time()

		init_sol = init_solution(n_vehicles, n_calls)
		init_list_sol = solution_to_list(init_sol,n_vehicles)
		cur_best_list = optimize_solution(init_list_sol, prob, cargo_dict, iterations, print_iter)
		cur_best = list_to_solution(cur_best_list)
		solutions.append(cur_best)
		avg_times.append(time() - start_time)

		if cost_function(cur_best, prob) < cost_function(best_solution, prob):
			best_solution = cur_best

		print(f"Done with test {test + 1}, Took {round(avg_times[test], 2)}s")
		print(f"cost: {cost_function(best_solution, prob)}")
		print(f"improvement: {100 * (cost_function(init_solution(n_vehicles, n_calls), prob) - cost_function(best_solution, prob)) / cost_function(init_solution(n_vehicles, n_calls), prob)}%")
		print(f"Sol: {best_solution}")

	return avg_times, solutions, best_solution

def main():
	num_tests = 10
	prob_load = 6
	total_iterations = 10000
	should_print_sol = True
	should_print_iter = True
	problems = ['Call_7_Vehicle_3.txt', 'Call_7_Vehicle_3.txt', 'Call_18_Vehicle_5.txt',
				'Call_35_Vehicle_7.txt', 'Call_80_Vehicle_20.txt', 'Call_130_Vehicle_40.txt',
				'Call_300_Vehicle_90.txt']
	if prob_load == 0:
		for i in range(1,len(problems)):
			print("Problem: ", problems[i], "test number: ", i)
			prob = init_problem(problems, i)
			avg_times, solutions, best_solution = run_tests(num_tests, total_iterations, prob, print_iter=should_print_iter)

			pprint(avg_times, solutions, prob, num_tests, problems, i, best_solution, print_best=should_print_sol, solve_method="Local Search")
	
	else:
		prob = init_problem(problems, prob_load)
		avg_times, solutions, best_solution = run_tests(num_tests, total_iterations, prob, print_iter=should_print_iter)
		pprint(avg_times, solutions, prob, num_tests, problems, prob_load, best_solution, solve_method="Local Search", print_best=should_print_sol, )

if __name__ == "__main__":
	main()

	# print(n_nodes)
	# print(n_vehicles)
	# print(n_calls)
	# print(Cargo)
	# print(TravelTime)
	# print(FirstTravelTime)
	# print(VesselCapacity)
	# print(LoadingTime)
	# print(UnloadingTime)
	# print(VesselCargo)
	# print(TravelCost)
	# print(FirstTravelCost)
	# print(PortCost)