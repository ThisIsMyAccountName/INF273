from time import time
from copy import deepcopy
from random import randint, shuffle, random, choices, seed
from pdp_utils.pdp_utils.Utils import cost_function, feasibility_check
from init import init_problem, init_solution, pprint
from math import e
from helpers import *
from collections import deque

seed(42)

def find_duplicated_cargo(solution, new_solution, problem, op_i, op_r):
	counts = {i:0 for i in range(1, problem['n_calls'] + 1)}
	for vessel in new_solution:
		for cargo in vessel:
			counts[cargo] += 1
	def missing_cargo(solution, n_calls):
		all_calls = set(range(1, n_calls + 1))
		in_sol = set([c for v in solution for c in v])
		return all_calls - in_sol
	dupes, missing = [k for k, v in counts.items() if v != 2], missing_cargo(new_solution, problem['n_calls'])
	if dupes or missing:
		iops = ["Greedy-First", "Greedy-Best", "RegretK-First", "RegretK-Best"]
		rops = ["Random", "Expensive", "Outsourced", "Expensive travel"]
		print(f"Found duplicates: {dupes} using remove with {rops[op_r]} and insert with {iops[op_i]}")
		print(f"Missing: {missing}")
		print("Old Solution: ", solution)
		print()
		print("Current Solution: ", new_solution)
		print("Feasibility: ", feasibility_check(list_to_solution(new_solution), problem))
		exit()

def weighted_sample_without_replacement(population, weights, k):
    v = [random() ** (1 / w) for w in weights]
    order = sorted(range(len(population)), key=lambda i: v[i])
    return [population[i] for i in order[-k:]]

def escape_local_minimum(solution, problem):
	# TODO: Not working correctly
	new_sol = deepcopy(solution)
	amount_to_move = randint(1, 3)
	# calls_to_move = sample(range(1, problem['n_calls'] + 1), amount_to_move)
	# ret_solution = [[c for c in v if c not in calls_to_move] for v in new_sol]
	ret_solution, calls_to_move, _ = remove_expensive_calls(solution, problem, amount_to_move)
	print(ret_solution, calls_to_move)
	ret_solution[-1].extend(calls_to_move)
	ret_solution[-1].extend(calls_to_move)
	return ret_solution


def optimize_solution(solution, problem, cargo_dict, iterations,  print_iter=False):
	n_calls = problem['n_calls']
	n_vehicles = problem['n_vehicles']
	current_iter = 0
	best_solution = solution
	best_cost = cost_function(list_to_solution(best_solution), problem)
	current_cost = best_cost
	non_improvement = 0
	no_best_found = 0

	found_best_at = 0
	times_moved = 0
	infeasable_solutions = 0
	# first is greedy-first, second is greedy-best, third is regretk-first, fourth is regretk-best
	op_weights_insert = [0.25, 0.25, 0.25, 0.25]
	op_counts_insert = [0] * len(op_weights_insert)
	# first is random, second is expensive, third is outsourced, fourth is expensive travel
	op_weights_remove = [0.25, 0.25, 0.25, 0.25]
	op_counts_remove = [0] * len(op_weights_remove)

	start_time = time()
	print_timer = time()

	while current_iter < iterations:
		if current_iter % (iterations // 10) == 0 and print_iter and current_iter != 0:
			print(f"Iteration: {current_iter} , Best cost: {cost_function(list_to_solution(best_solution), problem):,} found at {found_best_at}, times moved: {times_moved}, current cost: {cost_function(list_to_solution(current_solution), problem):,}, infeasable solutions: {infeasable_solutions}") #{current_solution, feasibility_check(list_to_solution(current_solution), problem)}")	
		
		#print solution every ~minute
		if print_timer + 60 < time():
			print()
			print()
			print(solution)
			print()
			print()
			print_timer = time()

		if current_iter % 100 == 0 and current_iter != 0:
			# print(f"Inserts: \n Greedy-First: {op_counts_insert[0]},\n Greedy_Best: {op_counts_insert[1]},\n RegretK-First: {op_counts_insert[2]},\n RegretK-Best: {op_counts_insert[3]}")
			# print(f"Removes: \n Random: {op_counts_remove[0]},\n Expensive: {op_counts_remove[1]},\n Outsourced: {op_counts_remove[2]},\n Expensive travel: {op_counts_remove[3]}")
			insert_sum = sum(op_counts_insert)
			remove_sum = sum(op_counts_remove)
			if insert_sum != 0 and remove_sum != 0:
				rate = 0.15
				for i in range(len(op_weights_insert)):
					op_weights_insert[i] = op_weights_insert[i] * (1 - rate) + rate * (op_counts_insert[i] / insert_sum)
				op_counts_insert= [0] * len(op_counts_insert)
				for i in range(len(op_weights_remove)):
					op_weights_remove[i] = op_weights_remove[i] * (1 - rate) + rate * (op_counts_remove[i] / remove_sum)
				op_counts_remove = [0] * len(op_counts_remove)
				print(f"Inserts: {[f'{weight:.2f}' for weight in op_weights_insert]}, \nRemoves: {[f'{weight:.2f}' for weight in op_weights_remove]}")
		
		if no_best_found > 1000:
			print("No better solution found in 1000 iterations")
			for i in range(n_vehicles):
				solution = escape_local_minimum(solution, problem)
			current_cost = cost_function(list_to_solution(solution), problem)
			non_improvement = 0
			no_best_found = 0
				
		if non_improvement > 150:
			print("Escaping local minimum")
			solution = escape_local_minimum(solution, problem)
			current_cost = cost_function(list_to_solution(solution), problem)
			non_improvement = 0


		current_solution, op_i, op_r = apply_remove_and_insert(solution, problem, op_weights_insert, op_weights_remove)
		
		# checks for dupe/missing cargo
		find_duplicated_cargo(current_solution, solution, problem, op_i, op_r)
		

		new_cost = cost_function(list_to_solution(current_solution), problem)
		if new_cost < current_cost:
			print(f"Best Cost: {best_cost:,}, New Cost: {new_cost:,}, it was {best_cost - new_cost:,} cost better, found at iteration: {current_iter}, time: {time() - start_time:.2f} seconds")
			# print(current_solution)
			if new_cost < best_cost:
				best_solution = deepcopy(current_solution)
				best_cost = new_cost
				op_counts_insert[op_i] += 4
				op_counts_remove[op_r] += 4
				found_best_at = current_iter
				no_best_found = 0
				non_improvement = 0
				
			else:
				op_counts_insert[op_i] += 1
				op_counts_remove[op_r] += 1

			current_cost = new_cost
			solution = current_solution

		else:
			non_improvement += 1
			no_best_found += 1

		current_iter += 1


	return best_solution, found_best_at

def apply_remove_and_insert(solution, problem, op_weights_insert, op_weights_remove):
	# backup = deepcopy(solution)
	# n = randint(1, problem['n_calls'] // 10 + 1)
	n = randint(1, 5)
	removed_solution, removed_calls, op_r = apply_removes(solution, problem, n, op_weights_remove)
	new_solution, op_i = apply_inserts(removed_solution, removed_calls, problem, n, op_weights_insert)
	return new_solution, op_i, op_r
	# removed_solution, removed_call, removed_from, op_r = apply_remove(solution, problem, op_weights_remove)
	# feasibility, new_solution, rest, op_i = apply_insert(removed_solution, removed_call, removed_from, problem, op_weights_insert)
	# if rest != None:
	# 	return False, backup, None, None
	# return feasibility, new_solution, op_i, op_r

def apply_removes(solution, problem, n, op_weights):
	# op_weights = [0.5,0,0.5,0]
	rand_float = random()
	if rand_float < op_weights[0]:
		return remove_random_calls(solution, problem, n)
	elif rand_float < sum(op_weights[0:2]):
		return remove_expensive_calls(solution, problem, n)
	elif rand_float < sum(op_weights[0:3]):
		return remove_outsourced_calls(solution, problem, n)
	elif rand_float < sum(op_weights[0:4]):
		return remove_expensive_travel_calls(solution, problem, n)
	else:
		return remove_random_calls(solution, problem, n)
	
def apply_inserts(solution, removed_calls, problem, n, op_weights):
	# op_weights = [1,0,0]
	rand_float = random()
	if rand_float < op_weights[0]:
		return insert_calls_greedy(solution, removed_calls, problem, "First")
	elif rand_float < sum(op_weights[0:2]):
		return regretk_inserts(solution, removed_calls, n, problem, "First")
	elif rand_float < sum(op_weights[0:3]):
		return insert_calls_greedy(solution, removed_calls, problem, "Best")
	elif rand_float < sum(op_weights[0:4]):
		return regretk_inserts(solution, removed_calls, n, problem, "Best")
	else:
		return insert_calls_greedy(solution, removed_calls, problem, "First")

def apply_remove(solution, problem, op_weights):
	rand_float = random()
	if rand_float < op_weights[0]:
		return remove_random_call(solution, problem)
	elif rand_float < sum(op_weights[0:2]):
		return remove_expensive_call(solution, problem)
	elif rand_float < sum(op_weights[0:3]):
		return remove_outsourced_call(solution, problem)
	elif rand_float < sum(op_weights[0:4]):
		return remove_expensive_travel_call(solution, problem)
	else:
		return remove_random_call(solution, problem)

def apply_insert(solution, cargo, from_vessel, problem, op_weights):
	rand_float = random()
	if rand_float < op_weights[0]:
		return insert_call_pick_best(solution, cargo, from_vessel, problem)
	elif rand_float < sum(op_weights[0:2]):
		return insert_call_pick_first(solution, cargo, from_vessel, problem)
	else:
		return insert_call_pick_first(solution, cargo, from_vessel, problem)

def regretk_inserts(removed_solution, removed_calls, k, problem, insert_type="First"):
	cargo_dict = problem['cargo_dict']

	best_posistions = {call: [] for call in removed_calls}
	for call in removed_calls:
		work_solution = deepcopy(removed_solution)
		for vessel_idx in cargo_dict[call]:
			regret_values = regretk_value_one_vessel(work_solution[vessel_idx], vessel_idx, call, problem)
			if regret_values:
				best_posistions[call].extend(regret_values)
	
	regret_values = {}
	inserts = {}
	for call, values in best_posistions.items():
		values.sort(key=lambda x: x[0])
		if values:
			if len(values) < k:
				regret_values[call] = float('inf')
			else:
				regret_values[call] = values[k-1][0] - values[0][0]
			inserts[call] = values[0]
	ordering = sorted(regret_values, key=lambda x: regret_values[x], reverse=True)
	for call in ordering:
		cost, i, j, vessel_idx = inserts[call]
		if insert_type == "First":
			success,  new_vessel = insert_call_greedy_first(vessel_idx, deepcopy(removed_solution[vessel_idx]), call, problem)
		elif insert_type == "Best":
			success,  new_vessel = insert_call_greedy_best(vessel_idx, deepcopy(removed_solution[vessel_idx]), call, problem)
		if not success:
			removed_solution[-1].append(call)
			removed_solution[-1].append(call)
		else:
			removed_solution[vessel_idx] = new_vessel
	
	# in case some calls gets lost
	all_calls = set(range(1, problem['n_calls'] + 1))
	in_sol = set([c for v in removed_solution for c in v])
	missing_calls = all_calls - in_sol
	# print("Missing calls: ", missing_calls)
	removed_solution[-1].extend(missing_calls)
	removed_solution[-1].extend(missing_calls)	
	##############################
	
	return removed_solution, 2 if insert_type == "First" else 3

def regretk_value_one_vessel(vessel, vessel_idx, call, problem):
	# print("Regret value")
	regret_values = []
	for i in range(len(vessel) + 1):
		pickup_temp = deepcopy(vessel)
		pickup_temp.insert(i, call)
		f1, c1 = feasability_one_vessel(pickup_temp, problem, vessel_idx, spesific_cargo=call)
		if f1 or c1 == "Vessel capacity exceeded":
			for j in range(i + 1, len(vessel) + 2):
				deliver_temp = deepcopy(pickup_temp)
				deliver_temp.insert(j, call)
				f2, c2 = feasability_one_vessel(deliver_temp, problem, vessel_idx, spesific_cargo=call)
				if f2:
					new_cost = cost_one_vessel(deliver_temp, problem, vessel_idx)
					regret_values.append((new_cost, i, j, vessel_idx))
				elif c2 == "Time window wrong for given cargo":
					break
		elif c1 == "Time window wrong for given cargo":
			break
	regret_values.sort(key=lambda x: x[0])
	return regret_values

def insert_calls_greedy(solution, calls_to_insert, problem, insert_type="First"):
	cargo_dict = problem['cargo_dict']
	ret_solution = solution
	shuffle(calls_to_insert)
	inserted = set()
	for call in calls_to_insert:
		if call in inserted:
			continue
		this_call_solution = deepcopy(ret_solution)
		best_cost = float('inf')
		vessels = list(cargo_dict[call])
		shuffle(vessels)
		for vessel_idx in vessels:
			vessel = this_call_solution[vessel_idx]
			if insert_type == "First":
				f, temp_vessel = insert_call_greedy_first(vessel_idx, vessel, call, problem)
			elif insert_type == "Best":
				f, temp_vessel = insert_call_greedy_best(vessel_idx, vessel, call, problem)
			if temp_vessel != vessel and f:
				new_solution = deepcopy(this_call_solution)
				new_solution[vessel_idx] = temp_vessel
				new_cost = cost_function(list_to_solution(new_solution), problem)
				if new_cost < best_cost:
					this_call_solution = deepcopy(new_solution)
					best_cost = new_cost
					inserted.add(call)
					break
		ret_solution = deepcopy(this_call_solution)
	all_calls = set(range(1, problem['n_calls'] + 1))
	in_sol = set([c for v in ret_solution for c in v])
	missing_calls = all_calls - in_sol
	ret_solution[-1].extend(missing_calls)
	ret_solution[-1].extend(missing_calls)
	# if missing_calls:
	# 	print("Missing calls: ", missing_calls)
	return ret_solution, 0 if insert_type == "First" else 1

def insert_call_greedy_first(vessel_idx, vessel, call, problem):
	# for i in range(len(vessel) + 1):
	# 	pickup_temp = deepcopy(vessel)
	# 	pickup_temp.insert(i, call)
	# 	f1, c1 = feasability_one_vessel(pickup_temp, problem, vessel_idx, spesific_cargo=call)
	# 	if f1 or c1 == "Vessel capacity exceeded":
	# 		for j in range(i, len(vessel) + 2):
	# 			deliver_temp = deepcopy(pickup_temp)
	# 			deliver_temp.insert(j, call)
	# 			f2, c2 = feasability_one_vessel(deliver_temp, problem, vessel_idx, spesific_cargo=call)
	# 			if f2:
	# 				return True, deliver_temp
	# 			elif c2 == "Time window wrong for given cargo":
	# 				break
	# 	elif c1 == "Time window wrong for given cargo":
	# 		break
	# return False, vessel
	for i in range(len(vessel) + 1):
		for j in range(i, len(vessel) + 2):
			new_list = deepcopy(vessel)
			new_list.insert(i, call)
			new_list.insert(j, call)
			f, c = feasability_one_vessel(new_list, problem, vessel_idx, spesific_cargo=call)
			if f:
				return True, new_list
	return False, vessel

def insert_call_greedy_best(vessel_idx, vessel, call, problem):
	# best_cost = float('inf')
	# best_list = vessel
	# for i in range(len(vessel) + 1):
	# 	pickup_temp = deepcopy(vessel)
	# 	pickup_temp.insert(i, call)
	# 	f1, c1 = feasability_one_vessel(pickup_temp, problem, vessel_idx, spesific_cargo=call)
	# 	if f1 or c1 == "Vessel capacity exceeded":
	# 		for j in range(i, len(vessel) + 2):
	# 			deliver_temp = deepcopy(pickup_temp)
	# 			deliver_temp.insert(j, call)
	# 			f2, c2 = feasability_one_vessel(deliver_temp, problem, vessel_idx, spesific_cargo=call)
	# 			if f2:
	# 				new_cost = cost_one_vessel(deliver_temp, problem, vessel_idx)
	# 				if new_cost < best_cost:
	# 					best_cost = new_cost
	# 					best_list = deepcopy(deliver_temp)
	# 			elif c2 == "Time window wrong for given cargo":
	# 				break
	# 	elif c1 == "Time window wrong for given cargo":
	# 		break
	# if best_cost == float('inf'):
	# 	return False, vessel
	# return True, best_list
	best_cost = float('inf')
	best_list = vessel
	for i in range(len(vessel) + 1):
		for j in range(i, len(vessel) + 2):
			new_list = deepcopy(vessel)
			new_list.insert(i, call)
			new_list.insert(j, call)
			f, c = feasability_one_vessel(new_list, problem, vessel_idx, spesific_cargo=call)
			if f:
				new_cost = cost_one_vessel(new_list, problem, vessel_idx)
				if new_cost < best_cost:
					best_cost = new_cost
					best_list = deepcopy(new_list)
	if best_cost == float('inf'):
		return False, vessel
	return True, best_list

def insert_call_pick_best(solution, cargo, from_vessel, problem):
	solution_copy = deepcopy(solution)
	cost = float('inf')
	cargo_dict = problem['cargo_dict']
	vessels = cargo_dict[cargo]
	shuffle(vessels)
	for vessel in vessels:
		if vessel == from_vessel:
			continue
		sol_copy = deepcopy(solution_copy)
		for i in range(len(sol_copy[vessel]) + 1):
			pickup_temp = deepcopy(sol_copy[vessel])
			pickup_temp.insert(i, cargo)
			f1,c1 = feasability_one_vessel(pickup_temp, problem, vessel, spesific_cargo=cargo)
			if f1 or c1 == "Vessel capacity exceeded":
				for j in range(i + 1, len(sol_copy[vessel]) + 2):
					deliver_temp = deepcopy(pickup_temp)
					deliver_temp.insert(j, cargo)
					f2, c2 = feasability_one_vessel(deliver_temp, problem, vessel, spesific_cargo=cargo)
					if f2:
						new_cost = cost_one_vessel(deliver_temp, problem, vessel)
						if cost == float('inf') or new_cost < cost:
							cost = new_cost
							best_solution = deliver_temp
							best_vehicle = vessel
							
					elif c2 == "Time window wrong for given cargo":
						break
			elif c1 == "Time window wrong for given cargo":
				break	
	if cost == float('inf'):
		return False, solution, cargo, 0
	sol_copy[best_vehicle] = best_solution
	return True, sol_copy, None, 0
	
def insert_call_pick_first(solution, cargo, from_vessel, problem):
	sol_copy = deepcopy(solution)
	cost = float('inf')
	cargo_dict = problem['cargo_dict']
	vessels = cargo_dict[cargo]
	shuffle(vessels)
	for vessel in vessels:
		if vessel == from_vessel:
			continue
		for i in range(len(sol_copy[vessel]) + 1):
			pickup_temp = deepcopy(sol_copy[vessel])
			pickup_temp.insert(i, cargo)
			f1, c1 = feasability_one_vessel(pickup_temp, problem, vessel, spesific_cargo=cargo)
			if f1 or c1 == "Vessel capacity exceeded":
				for j in range(i + 1, len(sol_copy[vessel]) + 2):
					deliver_temp = deepcopy(pickup_temp)
					deliver_temp.insert(j, cargo)
					f2, c2 = feasability_one_vessel(deliver_temp, problem, vessel, spesific_cargo=cargo)
					if f2:
						# new_cost = cost_one_vessel(deliver_temp, problem, vessel)
						# if new_cost < cost:
						# cost = new_cost
						sol_copy[vessel] = deliver_temp
						# print("Found better solution", new_cost)
						return True, sol_copy, None, 1
							
					elif c2 == "Time window wrong for given cargo":
						break
			elif c1 == "Time window wrong for given cargo":
				break
	return False, solution, cargo, 1

	sol_copy = deepcopy(solution)
	cargo_dict = problem['cargo_dict']
	for to_vessel in cargo_dict[cargo]:
		for i in range(len(sol_copy[to_vessel]) + 1):
			for j in range(i, len(sol_copy[to_vessel]) + 2):
				new_list = deepcopy(sol_copy[to_vessel])
				new_list.insert(i, cargo)
				new_list.insert(j, cargo)
				f, _ = feasability_one_vessel(new_list, problem, to_vessel)
				if f:
					sol_copy[to_vessel] = new_list
					return True, sol_copy, None, 1
	return False, solution, cargo, 1

def remove_random_call(solution, problem):
	calls = problem['n_calls']
	call = randint(1, calls)
	from_vessel = [i for i in range(len(solution)) if call in solution[i]][0]
	new_solution = [[c for c in v if c != call] for v in solution]
	return new_solution, call, from_vessel, 0

def remove_random_calls(solution, problem, n):
		n_calls = problem['n_calls']
		calls = sample(range(1, n_calls + 1), n)
		new_solution = [[c for c in v if c not in calls] for v in solution]
		return new_solution, calls, 0

def remove_expensive_travel_call(solution, problem):
	higest_cost = 0
	for i in range(len(solution)-1):
		if len(solution[i]) == 0:
			continue
		pre_cost = travel_cost_one_vessel(solution[i], problem, i)
		for cargo in solution[i]:
			temp_vessel = [c for c in solution[i] if c != cargo]
			post_cost = travel_cost_one_vessel(temp_vessel, problem, i)
			if pre_cost - post_cost > higest_cost:
				higest_cost = pre_cost - post_cost
				call = cargo
				from_vessel = i
	if higest_cost == 0:
		return remove_outsourced_call(solution, problem)
	new_solution = [[c for c in v if c != call] for v in solution]
	return new_solution, call, from_vessel, 3

def remove_expensive_travel_calls(solution, problem, n):
	probabilities = problem['probabilities']
	costs = set()
	for i in range(len(solution)-2):
		if len(solution[i]) == 0:
			continue
		pre_cost = travel_cost_one_vessel(solution[i], problem, i)
		for cargo in solution[i]:
			temp_vessel = [c for c in solution[i] if c != cargo]
			post_cost = travel_cost_one_vessel(temp_vessel, problem, i)
			costs.add((cargo, pre_cost - post_cost, i))
	costs = list(costs)
	if len(costs) < n:
		return remove_outsourced_calls(solution, problem, n)
	costs.sort(key=lambda x: x[1], reverse=True)
	probabilities = probabilities[:len(costs)]
	ws = [w / sum(probabilities) for w in probabilities]
	# print(len(costs) == len(ws), len(costs), len(ws))
	calls = choices([x[0] for x in costs], weights=ws, k=n)
	new_solution = [[c for c in v if c not in calls] for v in solution]
	return new_solution, calls, 3

def remove_expensive_call(solution, problem):
	higest_cost = 0
	for i in range(len(solution)-1):
		if len(solution[i]) == 0:
			continue
		pre_cost = cost_one_vessel(solution[i], problem, i)
		for cargo in solution[i]:
			temp_vessel = [c for c in solution[i] if c != cargo]
			post_cost = cost_one_vessel(temp_vessel, problem, i)
			if pre_cost - post_cost > higest_cost:
				higest_cost = pre_cost - post_cost
				call = cargo
				from_vessel = i
	if higest_cost == 0:
		return remove_outsourced_call(solution, problem)
	new_solution = [[c for c in v if c != call] for v in solution]
	return new_solution, call, from_vessel, 1

def remove_expensive_calls(solution, problem, n):
	Cargo = problem['Cargo']
	probabilities = problem['probabilities']
	costs = set()
	for i in range(len(solution)-1):
		if len(solution[i]) == 0:
			continue
		pre_cost = cost_one_vessel(solution[i], problem, i)
		for cargo in solution[i]:
			temp_vessel = [c for c in solution[i] if c != cargo]
			post_cost = cost_one_vessel(temp_vessel, problem, i)
			costs.add((cargo, pre_cost - post_cost, i))
	# costs.update([(j, Cargo[j - 1, 3] / 2, -1) for j in solution[-1]])
		
	costs = list(costs)
	if len(costs) == 0:
		return remove_outsourced_calls(solution, problem, n)
	costs.sort(key=lambda x: x[1], reverse=True)
	probabilities = probabilities[:len(costs)]
	ws = [w/sum(probabilities) for w in probabilities]
	# print(len(costs) == len(ws), len(costs), len(ws))
	calls = weighted_sample_without_replacement([x[0] for x in costs], ws, n)
	new_solution = [[c for c in v if c not in calls] for v in solution]
	return new_solution, calls, 1

def remove_outsourced_call(solution, problem):
	outsouced = solution[-1]
	if len(outsouced) == 0:
		return remove_random_call(solution, problem)
	call = sample(outsouced, 1)[0]
	new_solution = [[c for c in v if c != call] for v in solution]
	return new_solution, call, len(solution) - 1, 2

def remove_outsourced_calls(solution, problem, n):
	outsouced = solution[-1]
	if len(outsouced) < n:
		return remove_random_calls(solution, problem, n)
		# calls = outsouced
		# removed_solution, removed_calls, _ = remove_random_calls(solution, problem, n-len(calls))
		# removed_calls.extend(calls)
		# return removed_solution, removed_calls, 2
	calls = sample(outsouced, n)
	new_solution = [[c for c in v if c not in calls] for v in solution]
	return new_solution, calls, 2

def run_tests(num_tests, iterations, prob, print_iter=False):
	n_vehicles = prob['n_vehicles']
	n_calls = prob['n_calls']
	VesselCargo = prob['VesselCargo']
	avg_times, solutions = [], []
	cargo_dict = cargo_to_vessel(VesselCargo, n_calls)
	best_solution = init_solution(n_vehicles, n_calls)
	probabilities = [e**(-0.2*(x))-e**(-0.2*(x+1)) for x in range(n_calls)]
	problem = {
		'n_nodes': prob['n_nodes'],
		'n_vehicles': prob['n_vehicles'],
		'n_calls': prob['n_calls'],
		'Cargo': prob['Cargo'],
		'TravelTime': prob['TravelTime'],
		'FirstTravelTime': prob['FirstTravelTime'],
		'VesselCapacity': prob['VesselCapacity'],
		'LoadingTime': prob['LoadingTime'],
		'UnloadingTime': prob['UnloadingTime'],
		'VesselCargo': prob['VesselCargo'],
		'TravelCost': prob['TravelCost'],
		'FirstTravelCost': prob['FirstTravelCost'],
		'PortCost': prob['PortCost'],
		'cargo_dict': cargo_dict,
		'probabilities': probabilities
	}

	for test in range(num_tests):

		solution = solution_to_list(init_solution(n_vehicles, n_calls), n_vehicles)

		start_time = time()

		cur_best_list, found_best_at = optimize_solution(solution, problem, cargo_dict, iterations, print_iter)
		cur_best = list_to_solution(cur_best_list)
		solutions.append(cur_best)
		avg_times.append(time() - start_time)

		if cost_function(cur_best, prob) < cost_function(best_solution, prob):
			best_solution = cur_best

		print(f"Done with test {test + 1}, Took {round(avg_times[test], 2)}s")
		print(f"cost: {cost_function(cur_best, prob):,} Found on iteration: {found_best_at}")
		print(f"improvement: {round(100 * (cost_function(init_solution(n_vehicles, n_calls), prob) - cost_function(cur_best, prob)) / cost_function(init_solution(n_vehicles, n_calls), prob), 2)}%")
		print(f"Sol: {cur_best}")
		print()

	return avg_times, solutions, best_solution

def main():
	num_tests = 1
	prob_load = 2
	total_iterations = 1000
	should_print_sol = False
	should_print_iter = False
	problems = ['Call_7_Vehicle_3.txt', 'Call_7_Vehicle_3.txt', 'Call_18_Vehicle_5.txt',
				'Call_35_Vehicle_7.txt', 'Call_80_Vehicle_20.txt', 'Call_130_Vehicle_40.txt',
				'Call_300_Vehicle_90.txt']
	res = []
	if prob_load == 0:
		for i in range(1,len(problems)):
			print("Problem: ", problems[i], "test number: ", i)
			prob = init_problem(problems, i)
			avg_times, solutions, best_solution = run_tests(num_tests, total_iterations, prob, print_iter=False)
			res.append((avg_times, solutions, prob, num_tests, problems, i, best_solution, should_print_sol, "Simulated annealing 4ops"))

		for r in res:
			pprint(*r)
	else:
		prob = init_problem(problems, prob_load)
		avg_times, solutions, best_solution = run_tests(num_tests, total_iterations, prob, print_iter=should_print_iter)
		pprint(avg_times, solutions, prob, num_tests, problems, prob_load, best_solution, solve_method="Simulated annealing", print_best=should_print_sol)

if __name__ == "__main__":
		main()