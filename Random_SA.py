from time import time
from copy import deepcopy
from random import sample, randint, shuffle, random
from collections import defaultdict
from pdp_utils.pdp_utils.Utils import feasibility_check, cost_function
from init import init_problem, init_solution, pprint
import numpy as np
from helpers import *

def pick_random_cargo(list_sol):
	picks = [i for i in range(len(list_sol)) if len(list_sol[i]) > 2]
	vessel = sample(picks, 1)[0]
	cargo = sample(list_sol[vessel], 1)[0]
	# print(cargo, vessel, "random")
	return cargo, vessel, []

def find_cargo_to_move(list_solution, prob, memo, poss_moves):
	while poss_moves:
		_, cargo_to_move, move_from  = poss_moves.pop()
		if cargo_to_move in list_solution[move_from]:
			return cargo_to_move, move_from, poss_moves
	
	return find_most_expensive_cargo_to_move(list_solution, prob, memo)

def find_most_expensive_cargo_to_move(list_solution, prob, memo):
	Cargo = prob['Cargo']
	lists = deepcopy(list_solution[:-1])
	costs = []
	for i in range(len(lists)):
		if len(lists[i]) <= 2:
			continue
		current = minus_one(lists[i])
		cargo_in_ves = set(current)
		for j in cargo_in_ves:
			v1 = deepcopy(current)
			pre_cost = cost_one_vessel(v1, prob, i)
			v1.remove(j)
			v1.remove(j)
			post_cost = cost_one_vessel(v1, prob, i)
			costs.append((pre_cost - post_cost, j+1 , i))

	last_vessel = len(list_solution) - 1
	costs.extend([(Cargo[j, 3], j + 1, last_vessel) for j in minus_one(list_solution[-1])])

	most_expensive = sorted(costs, key=lambda x: x[0])

	return most_expensive[-1][1], most_expensive[-1][2], most_expensive




def optimize_solution(org_solution, prob, cargo_dict, temperature, a, iterations=200, print_iter=False):
	incumbent = deepcopy(org_solution)
	best_solution = org_solution
	best_cost = cost_function(list_to_solution(best_solution), prob)
	memo = set()
	n_vehicles = prob['n_vehicles']
	n_calls = prob['n_calls']
	non_impoving = 0
	times_moved = 0
	poss_moves = []
	infeasable_solutions = 0
	iter = 0
	while iter < iterations:
		if iter % (iterations // 10) == 0 and print_iter:
			print(f"Iteration: {iter} Temperature: {int(temperature):,}, Test cost: {cost_function(list_to_solution(best_solution), prob):,}, times moved: {times_moved}, current cost: {cost_function(list_to_solution(incumbent), prob):,}, infeasable solutions: {infeasable_solutions}")

		# possibilities = copy(cargo_dict[something])
		# try next possibility
			
		# feasebility = False
		# while not feasebility:
		# 	incumbent_copy = deepcopy(incumbent)
		# 	cargo_to_move, move_from, poss_moves = find_cargo_to_move(incumbent_copy, prob, memo, poss_moves)
		# 	remove_elem_from(incumbent_copy, cargo_to_move, move_from)
		# 	feasebility, new_sol = insert_randomly(incumbent_copy, cargo_dict[cargo_to_move], move_from, cargo_to_move, prob)
		
		# old_cost = cost_function(list_to_solution(incumbent), prob)
		# new_cost = cost_function(list_to_solution(new_sol), prob)
		# E = new_cost - old_cost


		incumbent_copy = deepcopy(incumbent)
		cargo_to_move, move_from, _ = pick_random_cargo(incumbent_copy)
		# cargo_to_move, move_from, poss_moves = find_cargo_to_move(incumbent_copy, prob, memo, poss_moves)
		remove_elem_from(incumbent_copy, cargo_to_move, move_from)
		move_to = sample(cargo_dict[cargo_to_move], 1)[0]
		feasebility, new_sol = insert_number_at_all_positions_pick_first(incumbent_copy, move_to, move_from, cargo_to_move, prob)
		old_cost = cost_function(list_to_solution(incumbent), prob)
		new_cost = cost_function(list_to_solution(new_sol), prob)
		E = new_cost - old_cost

		if not feasebility:
			infeasable_solutions += 1

		if feasebility and E < 0:
			incumbent = new_sol
			incum_cost = new_cost
			non_impoving = 0
			if incum_cost < best_cost:
				if print_iter and iter > (iterations // 10):
					print(f"Found new best solution cost: {incum_cost:,}, it was {best_cost - incum_cost:,} cost better")
				best_solution = incumbent
				best_cost = incum_cost

		elif feasebility and random() < np.exp( -E / temperature):
			# print("accepting worse solution", E, "with probability", round(np.exp(-E / temperature), 10))
			incumbent = new_sol

			# need to tinker here
			# non_impoving -= 1
		else:
			non_impoving += 1
			# print("rejecting worse solution", E, "with probability", round(np.exp(-E / temperature), 10), new_sol)
		scaler = randint(1, n_vehicles // 10 + 2)
		if non_impoving > n_calls * scaler:
			# print(f"Moving {scaler} cargo to outsource")
			incumbent = Move_x_to_outsource(incumbent, scaler, prob, memo)
			times_moved += scaler
			non_impoving = 0

			iter += scaler
			temperature *= a
			continue

		temperature *= a
		iter += 1

	return best_solution

def insert_randomly(incumbent, move_tos, move_from, cargo_to_move, prob):
	feasability = False
	sol_copy = deepcopy(incumbent)
	shuffle(move_tos)
	for move_to in move_tos:
		pickup_range = list(range(0, len(sol_copy[move_to]) + 1))
		dropoff_range = list(range(0, len(sol_copy[move_to]) + 2))
		shuffle(pickup_range)
		shuffle(dropoff_range)
		for i in pickup_range:
			for j in dropoff_range:
				new_list = deepcopy(sol_copy[move_to])
				new_list.insert(i, cargo_to_move)
				new_list.insert(j, cargo_to_move)
				if feasability_one_vessel(new_list, prob, move_to)[0]:
					sol_copy[move_to] = new_list
					return True, sol_copy
	add_elem_to(sol_copy, cargo_to_move, move_from)
	return False, sol_copy





def warmup_solution(org_solution, prob, cargo_dict, iterations=100):
	incumbent_sol =  org_solution
	best_solution =  org_solution
	E_acum = 0
	for _ in range(iterations):
		incumbent_sol1 =  deepcopy(incumbent_sol)
		cargo_to_move, move_from, _ = pick_random_cargo(incumbent_sol1)
		# cargo_to_move, move_from = find_cargo_to_move(incumbent_sol1, prob, set())

		remove_elem_from(incumbent_sol1, cargo_to_move, move_from)
		feasibility, new_sol = insert_randomly(incumbent_sol1, cargo_dict[cargo_to_move], move_from, cargo_to_move, prob)
		old_cost = cost_function(list_to_solution(incumbent_sol), prob)
		new_cost = cost_function(list_to_solution(new_sol), prob)
		E = new_cost - old_cost
		# print(incumbent_sol)
		# print(new_sol)
		# print(sum(sum(x) for x in incumbent_sol) == sum(sum(x) for x in new_sol) == sum(init_solution(prob['n_vehicles'], prob['n_calls'])), feasibility)
		# print(E)

		if feasibility and E < 0:
			incumbent_sol =  new_sol
			incumbent_cost = new_cost
			if incumbent_cost < cost_function(list_to_solution(best_solution), prob):
				best_solution =  incumbent_sol
		elif feasibility:
			if random() < 0.8:
				incumbent_sol =  new_sol
			E_acum += E	

	return E_acum / iterations

def run_tests(num_tests, iterations, warmup, prob, print_iter=False):

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
	cargo_dict = cargo_to_vessel(VesselCargo, n_calls)
	best_solution = init_solution(n_vehicles, n_calls)

	for test in range(num_tests):

		incumbent = solution_to_list(init_solution(n_vehicles, n_calls), n_vehicles)

		start_time = time()

		warmup_iterations = warmup

		delta_avg = warmup_solution(incumbent, prob, cargo_dict, iterations=warmup_iterations)
		if not delta_avg:
			temperature = 100000
			a = 0.999
			print("didn't find a detla_avg")
		else:
			T0 = (-delta_avg) / np.log(0.8)
			Tf = 0.1
			a = (Tf / T0) ** (1 / iterations)
			temperature = T0
		incumbent = solution_to_list(init_solution(n_vehicles, n_calls), n_vehicles)
		cur_best_list = optimize_solution(incumbent, prob, cargo_dict, temperature, a, iterations, print_iter)
		cur_best = list_to_solution(cur_best_list)
		solutions.append(cur_best)
		avg_times.append(time() - start_time)

		if cost_function(cur_best, prob) < cost_function(best_solution, prob):
			best_solution = cur_best

		print(f"Done with test {test + 1}, Took {round(avg_times[test], 2)}s")
		print(f"cost: {cost_function(cur_best, prob):,}")
		print(f"improvement: {round(100 * (cost_function(init_solution(n_vehicles, n_calls), prob) - cost_function(cur_best, prob)) / cost_function(init_solution(n_vehicles, n_calls), prob), 2)}%")
		print(f"Sol: {cur_best}")
		# print(f"Improvement: {100 * (cost_function(init_sol, prob) - cost_function(cur_best, prob)) / cost_function(init_sol, prob)}%")
		# print("feasibility: ", feasibility_check(best_solution, prob))
		# print(sum(best_solution), sum(init_sol))

	return avg_times, solutions, best_solution

def main():
	num_tests = 1
	prob_load = 1
	warmup = 100
	total_iterations = 10000 - warmup
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
			avg_times, solutions, best_solution = run_tests(num_tests, total_iterations, warmup, prob, print_iter=should_print_iter)
			res.append((avg_times, solutions, prob, num_tests, problems, i, best_solution, should_print_sol, "Simulated annealing"))

		for r in res:
			pprint(*r)
	else:
		prob = init_problem(problems, prob_load)
		avg_times, solutions, best_solution = run_tests(num_tests, total_iterations, warmup, prob, print_iter=should_print_iter)
		pprint(avg_times, solutions, prob, num_tests, problems, prob_load, best_solution, solve_method="Simulated annealing", print_best=should_print_sol)

if __name__ == "__main__":
	main()