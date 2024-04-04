from time import time
from copy import deepcopy
from random import sample, randint, shuffle, random
from collections import defaultdict
from pdp_utils.pdp_utils.Utils import feasibility_check, cost_function
from init import init_problem, init_solution, pprint
import numpy as np
from helpers import *

def pick_random_cargo(list_sol):
	if list_sol[-1] and random() < 0.8:
		cargo = sample(list_sol[-1], 1)[0]
		return cargo, len(list_sol) - 1, []
	picks = [i for i in range(len(list_sol)) if len(list_sol[i]) > 2]
	vessel = sample(picks, 1)[0]
	cargo = sample(list_sol[vessel], 1)[0]
	# print(cargo, vessel, "random")
	return cargo, vessel, []

def find_cargo_to_move(list_solution, prob, poss_moves):
	while poss_moves:
		_, cargo_to_move, move_from  = poss_moves.pop()
		if cargo_to_move in list_solution[move_from]:
			return cargo_to_move, move_from, poss_moves
	
	return find_most_expensive_cargo_to_move(list_solution, prob)

def find_most_expensive_cargo_to_move(list_solution, prob):
	Cargo = prob['Cargo']
	lists = list_solution
	last_vessel = len(list_solution) - 1

	costs = []
	for i in range(len(lists)-1):
		if len(lists[i]) <= 2:
			continue
		current = minus_one(lists[i])
		cargo_in_ves = set(current)
		pre_cost = cost_one_vessel(current, prob, i)
		for j in cargo_in_ves:
			v1 = deepcopy(current)
			v1.remove(j)
			v1.remove(j)
			post_cost = cost_one_vessel(v1, prob, i)
			if pre_cost - post_cost > 0:
				costs.append((pre_cost - post_cost, j+1 , i))


	costs.extend([(Cargo[j - 1, 3] / 2, j, last_vessel) for j in list_solution[-1]])

	most_expensive = sorted(costs, key=lambda x: x[0])
	_, cargo_to_move, move_from = most_expensive.pop()
	return cargo_to_move, move_from, most_expensive

def cheepest_insertion(list_sol, prob, compatible_vessels, cargo_to_move):
	copy = deepcopy(list_sol)
	r = []
	for i in compatible_vessels:
		vessel = deepcopy(copy[i])
		vessel.append(cargo_to_move)
		vessel.append(cargo_to_move)
		cost = cost_one_vessel(vessel, prob, i)
		r.append((cost, i))

	r.sort(key=lambda x: x[0])
	return r[0][1]

def optimize_solution(org_solution, prob, cargo_dict, feas_memo, temperature, a, iterations=200, print_iter=False):
	incumbent = deepcopy(org_solution)
	best_solution = org_solution
	best_cost = float('inf')
	n_vehicles = prob['n_vehicles']
	n_calls = prob['n_calls']
	non_impoving = 0
	times_moved = 0
	poss_moves = []
	infeasable_solutions = 0
	found_best_at = 0
	iter = 0
	old_cost = cost_function(list_to_solution(incumbent), prob)
	while iter < iterations:
		if iter % (iterations // 10) == 0 and print_iter:
			print(f"Iteration: {iter} Temperature: {int(temperature):,}, Best cost: {cost_function(list_to_solution(best_solution), prob):,} found at {found_best_at}, times moved: {times_moved}, current cost: {cost_function(list_to_solution(incumbent), prob):,}, infeasable solutions: {infeasable_solutions}")
		
		scaler = randint(1, 3)
		if non_impoving > n_calls * scaler:
			print("Moving to outsource")
			incumbent = Move_x_to_outsource(incumbent, scaler, prob)
			times_moved += scaler
			non_impoving = 0
			continue

		incumbent_copy = deepcopy(incumbent)
		# apply operator
		feasability, new_sol, poss_moves = make_move(incumbent_copy, prob, cargo_dict, feas_memo, poss_moves)

		new_cost = cost_function(list_to_solution(new_sol), prob)
		E = new_cost - old_cost
		old_cost = new_cost

		if feasability and E < 0:
			incumbent = new_sol
			incum_cost = new_cost
			non_impoving = 0
			if incum_cost < best_cost:
				if print_iter and iter > (iterations // 10):
					print(f"Found new best solution cost: {incum_cost:,}, it was {best_cost - incum_cost:,} cost better, found at iteration: {iter}")
				best_solution = incumbent
				best_cost = incum_cost
				found_best_at = iter

		elif feasability and random() < np.exp( -E / temperature):
			incumbent = new_sol

		else:
			non_impoving += 1
			infeasable_solutions += 1

		temperature *= a
		iter += 1

	return best_solution

def make_move(incumbent_copy, prob, cargo_dict, feas_memo, poss_moves):
	choice = randint(1, 5)
	choice = 1
	if choice == 1:
		cargo_to_move, move_from, poss_moves = find_cargo_to_move(incumbent_copy, prob, poss_moves)
		remove_elem_from(incumbent_copy, cargo_to_move, move_from)
		move_to = sample(cargo_dict[cargo_to_move], 1)[0]
		feasebility, new_sol = insert_number_at_all_positions_pick_first(incumbent_copy, move_to, move_from, cargo_to_move, prob, feas_memo)
	elif choice == 2:
		cargo_to_move, move_from, poss_moves = pick_random_cargo(incumbent_copy)
		remove_elem_from(incumbent_copy, cargo_to_move, move_from)
		move_to = sample(cargo_dict[cargo_to_move], 1)[0]
		feasebility, new_sol = insert_number_at_all_positions_pick_first(incumbent_copy, move_to, move_from, cargo_to_move, prob, feas_memo)
	elif choice == 3:
		cargo_to_move, move_from, poss_moves = find_cargo_to_move(incumbent_copy, prob, poss_moves)
		remove_elem_from(incumbent_copy, cargo_to_move, move_from)
		move_to = sample(cargo_dict[cargo_to_move], 1)[0]
		feasebility, new_sol = insert_number_at_all_positions_pick_best(incumbent_copy, move_to, move_from, cargo_to_move, prob, feas_memo)
	elif choice == 4:
		cargo_to_move, move_from, poss_moves = pick_random_cargo(incumbent_copy)
		remove_elem_from(incumbent_copy, cargo_to_move, move_from)
		move_to = sample(cargo_dict[cargo_to_move], 1)[0]
		feasebility, new_sol = insert_number_at_all_positions_pick_random(incumbent_copy, move_to, move_from, cargo_to_move, prob, feas_memo)
	elif choice == 5:
		cargo_to_move, move_from, poss_moves = find_cargo_to_move(incumbent_copy, prob, poss_moves)
		remove_elem_from(incumbent_copy, cargo_to_move, move_from)
		move_to = sample(cargo_dict[cargo_to_move], 1)[0]
		feasebility, new_sol = insert_number_at_all_positions_pick_random(incumbent_copy, move_to, move_from, cargo_to_move, prob, feas_memo)
	
	return feasebility, new_sol, poss_moves


def warmup_solution(org_solution, prob, cargo_dict, feas_memo, iterations=100):
	incumbent_sol = org_solution.copy()
	best_cost = cost_function(list_to_solution(org_solution), prob)
	E_acum = 0
	for _ in range(iterations):
		incumbent_sol1 =  deepcopy(incumbent_sol)
		feasability, new_sol, _ = make_move(incumbent_sol1, prob, cargo_dict, feas_memo, [])
		new_cost = cost_function(list_to_solution(new_sol), prob)
		E = new_cost - best_cost
		if feasability and E < 0:
			incumbent_sol = new_sol
			if new_cost < best_cost:
				best_cost = new_cost
		elif feasability:
			if random() < 0.8:
				incumbent_sol = new_sol
			E_acum += E	
	return E_acum / iterations, incumbent_sol

def run_tests(num_tests, iterations, warmup, prob, print_iter=False, feasability_memo=set()):

	n_nodes = prob['n_nodes']
	n_vehicles = prob['n_vehicles']
	n_calls = prob['n_calls']
	VesselCargo = prob['VesselCargo']
	avg_times, solutions = [], []
	cargo_dict = cargo_to_vessel(VesselCargo, n_calls)
	best_solution = init_solution(n_vehicles, n_calls)

	for test in range(num_tests):

		incumbent = solution_to_list(init_solution(n_vehicles, n_calls), n_vehicles)

		start_time = time()

		warmup_iterations = warmup

		delta_avg, incumbent = warmup_solution(incumbent, prob, cargo_dict, feasability_memo, iterations=warmup_iterations)
		if not delta_avg:
			temperature = 100000
			a = 0.999
			print("didn't find a detla_avg")
		else:
			T0 = (-delta_avg) / np.log(0.8)
			Tf = 0.1
			a = (Tf / T0) ** (1 / iterations)
			temperature = T0
		# incumbent = solution_to_list(init_solution(n_vehicles, n_calls), n_vehicles)
		cur_best_list = optimize_solution(incumbent, prob, cargo_dict, feasability_memo, temperature, a, iterations, print_iter)
		cur_best = list_to_solution(cur_best_list)
		solutions.append(cur_best)
		avg_times.append(time() - start_time)

		if cost_function(cur_best, prob) < cost_function(best_solution, prob):
			best_solution = cur_best

		print(f"Done with test {test + 1}, Took {round(avg_times[test], 2)}s")
		print(f"cost: {cost_function(cur_best, prob):,}")
		print(f"improvement: {round(100 * (cost_function(init_solution(n_vehicles, n_calls), prob) - cost_function(cur_best, prob)) / cost_function(init_solution(n_vehicles, n_calls), prob), 2)}%")
		print(f"Sol: {cur_best}")
		print()

	return avg_times, solutions, best_solution

def main():
	num_tests = 1
	prob_load = 3
	warmup = 100
	total_iterations = 1000 - warmup
	should_print_sol = True
	should_print_iter = True
	problems = ['Call_7_Vehicle_3.txt', 'Call_7_Vehicle_3.txt', 'Call_18_Vehicle_5.txt',
				'Call_35_Vehicle_7.txt', 'Call_80_Vehicle_20.txt', 'Call_130_Vehicle_40.txt',
				'Call_300_Vehicle_90.txt']
	res = []
	if prob_load == 0:
		for i in range(1,len(problems)):
			print("Problem: ", problems[i], "test number: ", i)
			prob = init_problem(problems, i)
			avg_times, solutions, best_solution = run_tests(num_tests, total_iterations, warmup, prob, print_iter=False)
			res.append((avg_times, solutions, prob, num_tests, problems, i, best_solution, should_print_sol, "Simulated annealing"))

		for r in res:
			pprint(*r)
	else:
		prob = init_problem(problems, prob_load)
		avg_times, solutions, best_solution = run_tests(num_tests, total_iterations, warmup, prob, print_iter=should_print_iter)
		pprint(avg_times, solutions, prob, num_tests, problems, prob_load, best_solution, solve_method="Simulated annealing", print_best=should_print_sol)

if __name__ == "__main__":
	main()