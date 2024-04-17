from time import time
from copy import deepcopy
from random import randint, shuffle, random, seed
from pdp_utils.pdp_utils.Utils import cost_function, feasibility_check
from init import init_problem, init_solution, pprint
from math import e
from helpers import *
import cProfile

seed(42)

def find_duplicated_cargo(solution, new_solution, problem):
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
		print(f"Missing: {missing}")
		print("Old Solution: ", solution)
		print()
		print("Current Solution: ", new_solution)
		print("Feasibility: ", feasibility_check(list_to_solution(new_solution), problem))
		exit()

#stole online so i dont need np.choices >:(
def weighted_sample_without_replacement(population, weights, k):
	v = [random() ** (1 / w) for w in weights]
	order = sorted(range(len(population)), key=lambda i: v[i])
	return [population[i] for i in order[-k:]]

def escape_local_minimum(solution, problem, ops):
	amount_to_move = 10
	for _ in range(amount_to_move):
		new_solution, *_ = make_move(solution, problem, [1] + [0] * (ops - 1))
		if feasibility_check(list_to_solution(new_solution), problem):
			solution = new_solution
	return solution
	
def print_weights(weights, counts):
	for i, w in enumerate(weights):
		print(f"{i+1}: {w:.2f}: {counts[i]:,}")
	print()


def optimize_solution(solution, problem, iterations,  print_iter=False):
	opDeltas = problem['opDeltas']
	current_iter = 0
	best_solution = solution
	best_cost = cost_function(list_to_solution(best_solution), problem)
	current_cost = best_cost
	non_improvement = 0

	found_best_at = 0
	times_moved = 0
	infeasable_solutions = 0

	ops = 5
	op_weights = [1 / ops] * ops
	op_counts = [0] * len(op_weights)
	op_weights_list = [op_weights]

	start_time = time()
	print_timer = time()
	print_c = 0
	while current_iter < iterations:
		if current_iter % (iterations // 10) == 0 and print_iter and current_iter != 0:
			print(f"Iteration: {current_iter} , Best cost: {cost_function(list_to_solution(best_solution), problem):,} found at {found_best_at}, times moved: {times_moved}, current cost: {cost_function(list_to_solution(current_solution), problem):,}, infeasable solutions: {infeasable_solutions}") #{current_solution, feasibility_check(list_to_solution(current_solution), problem)}")	
		
		#print solution every ~minute
		if print_timer + 10 < time() and False:
			if print_c >= 6:
				print()
				print()
				print(solution)
				print()
				print()
				print_c = 0
			print(f"{int(time() - start_time)} seconds, on iteration: {current_iter}")
			print_timer = time()
			print_c += 1

		if current_iter % 100 == 0 and current_iter != 0:
			insert_sum = sum(op_counts)
			if insert_sum != 0:
				rate = 0.2
				for i in range(len(op_weights)):
					op_weights[i] = op_weights[i] * (1 - rate) + rate * (op_counts[i] / insert_sum)
				op_counts = [1] * len(op_counts)
				op_weights_list.append(deepcopy(op_weights))
			
		if non_improvement > 500:
			solution = escape_local_minimum(solution, problem, ops)
			current_cost = cost_function(list_to_solution(solution), problem)
			non_improvement = 0


		current_solution, op = make_move(solution, problem, op_weights)
		
		find_duplicated_cargo(solution, current_solution, problem)

		# 0.05 good for case 2
		D = 0.1 * ((iterations - current_iter) / iterations) * best_cost
		new_cost = cost_function(list_to_solution(current_solution), problem)
		if new_cost < current_cost:
			print(f"Best Cost: {best_cost:,}, Found at: {found_best_at} New Cost: {new_cost:,}, it was {best_cost - new_cost:,} cost better, found at iteration: {current_iter}, time: {time() - start_time:.2f} seconds, D: {int(D):,}")
			if new_cost < best_cost:
				best_solution = deepcopy(current_solution)
				opDeltas[op].append((current_iter, new_cost - best_cost))
				best_cost = new_cost
				op_counts[op] += 4
				found_best_at = current_iter
				non_improvement = 0
				
			else:
				op_counts[op] += 2
			
			solution = current_solution
			current_cost = new_cost

		elif new_cost < best_cost + D:
			solution = current_solution
			current_cost = new_cost

		else:
			non_improvement += 1

		current_iter += 1

	print(op_weights_list)

	return best_solution, found_best_at



def make_move(solution, problem, op_weights):
	rand_float = random()
	if rand_float < op_weights[0]:
		return remove_random_insert_greedy_best(solution, problem), 0
	elif rand_float < sum(op_weights[0:2]):
		return remove_similar_insert_greedy_best(solution, problem), 1
	elif rand_float < sum(op_weights[0:3]):
		return remove_outsourced_insert_greedy_first(solution, problem), 2
	elif rand_float < sum(op_weights[0:4]):
		return remove_expensive_insert_greedy_first(solution, problem), 3
	elif rand_float < sum(op_weights[0:5]):
		return remove_expensive_travel_insert_greedy_first(solution, problem), 4
	else:
		return remove_random_insert_greedy_first(solution, problem), 0
	
f = 3
t = 6
def insert_op(veh_idx, vessel, call, problem, insert_type):
	if insert_type == "First":
		return insert_call_greedy_first(veh_idx, vessel, call, problem)
	elif insert_type == "Best":
		return insert_call_greedy_best(veh_idx, vessel, call, problem)
	else:
		return insert_call_greedy_first(veh_idx, vessel, call, problem)
def remove_random_insert_greedy_first(solution, problem):
	removed_solution, removed_calls = remove_random_calls(solution, problem, randint(f, t))
	new_solution = insert_calls_greedy(removed_solution, removed_calls, problem, "First")
	return new_solution
def remove_expensive_insert_greedy_first(solution, problem):
	removed_solution, removed_calls = remove_expensive_calls(solution, problem, randint(f, t))
	new_solution = insert_calls_greedy(removed_solution, removed_calls, problem, "First")
	return new_solution
def remove_outsourced_insert_greedy_first(solution, problem):
	removed_solution, removed_calls = remove_outsourced_calls(solution, problem, randint(f, t))
	new_solution = insert_calls_greedy(removed_solution, removed_calls, problem, "First")
	return new_solution
def remove_expensive_travel_insert_greedy_first(solution, problem):
	removed_solution, removed_calls = remove_expensive_travel_calls(solution, problem, randint(f, t))
	new_solution = insert_calls_greedy(removed_solution, removed_calls, problem, "First")
	return new_solution
def remove_random_insert_greedy_best(solution, problem):
	removed_solution, removed_calls = remove_random_calls(solution, problem, randint(f, t))
	new_solution = insert_calls_greedy(removed_solution, removed_calls, problem, "Best")
	return new_solution
def remove_expensive_insert_greedy_best(solution, problem):
	removed_solution, removed_calls = remove_expensive_calls(solution, problem, randint(f, t))
	new_solution = insert_calls_greedy(removed_solution, removed_calls, problem, "Best")
	return new_solution
def remove_outsourced_insert_greedy_best(solution, problem):
	removed_solution, removed_calls = remove_outsourced_calls(solution, problem, randint(f, t))
	new_solution = insert_calls_greedy(removed_solution, removed_calls, problem, "Best")
	return new_solution
def remove_expensive_travel_insert_greedy_best(solution, problem):
	removed_solution, removed_calls = remove_expensive_travel_calls(solution, problem, randint(f, t))
	new_solution = insert_calls_greedy(removed_solution, removed_calls, problem, "Best")
	return new_solution
def remove_random_insert_regretk_first(solution, problem):
	n = randint(f, t)
	k = randint(2, 4)
	removed_solution, removed_calls = remove_random_calls(solution, problem, n)
	new_solution = regretk_inserts(removed_solution, removed_calls, k, problem, "First")
	return new_solution
def remove_expensive_insert_regretk_first(solution, problem):
	n = randint(f, t)
	k = randint(2, 4)
	removed_solution, removed_calls = remove_expensive_calls(solution, problem, n)
	new_solution = regretk_inserts(removed_solution, removed_calls, k, problem, "First")
	return new_solution
def remove_outsourced_insert_regretk_first(solution, problem):
	n = randint(f, t)
	k = randint(2, 4)
	removed_solution, removed_calls = remove_outsourced_calls(solution, problem, n)
	new_solution = regretk_inserts(removed_solution, removed_calls, k, problem, "First")
	return new_solution
def remove_expensive_travel_insert_regretk_first(solution, problem):
	n = randint(f, t)
	k = randint(2, 4)
	removed_solution, removed_calls = remove_expensive_travel_calls(solution, problem, n)
	new_solution = regretk_inserts(removed_solution, removed_calls, k, problem, "First")
	return new_solution
def remove_random_insert_regretk_best(solution, problem):
	n = randint(f, t)
	k = randint(2, 4)
	removed_solution, removed_calls = remove_random_calls(solution, problem, n)
	new_solution = regretk_inserts(removed_solution, removed_calls, k, problem, "Best")
	return new_solution
def remove_expensive_insert_regretk_best(solution, problem):
	n = randint(f, t)
	k = randint(2, 4)
	removed_solution, removed_calls = remove_expensive_calls(solution, problem, n)
	new_solution = regretk_inserts(removed_solution, removed_calls, k, problem, "Best")
	return new_solution
def remove_outsourced_insert_regretk_best(solution, problem):
	n = randint(f, t)
	k = randint(2, 4)
	removed_solution, removed_calls = remove_outsourced_calls(solution, problem, n)
	new_solution = regretk_inserts(removed_solution, removed_calls, k, problem, "Best")
	return new_solution	
def remove_expensive_travel_insert_regretk_best(solution, problem):
	n = randint(f, t)
	k = randint(2, 4)
	removed_solution, removed_calls = remove_expensive_travel_calls(solution, problem, n)
	new_solution = regretk_inserts(removed_solution, removed_calls, k, problem, "Best")
	return new_solution
def remove_similar_insert_greedy_first(solution, problem):
	removed_solution, removed_calls = remove_similar_calls(solution, problem, randint(f, t))
	new_solution = insert_calls_greedy(removed_solution, removed_calls, problem, "First")
	return new_solution
def remove_similar_insert_greedy_best(solution, problem):
	removed_solution, removed_calls = remove_similar_calls(solution, problem, randint(f, t))
	new_solution = insert_calls_greedy(removed_solution, removed_calls, problem, "Best")
	return new_solution
def remove_similar_insert_regretk_first(solution, problem):
	n = randint(f, t)
	k = randint(2, 4)
	removed_solution, removed_calls = remove_similar_calls(solution, problem, n)
	new_solution = regretk_inserts(removed_solution, removed_calls, k, problem, "First")
	return new_solution
def remove_similar_insert_regretk_best(solution, problem):
	n = randint(f, t)
	k = randint(2, 4)
	removed_solution, removed_calls = remove_similar_calls(solution, problem, n)
	new_solution = regretk_inserts(removed_solution, removed_calls, k, problem, "Best")
	return new_solution
def apply_removes(solution, problem, n, op_weights):
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
	# 2swap opperator. based on cost, % of full cargo, 
	else:
		return insert_calls_greedy(solution, removed_calls, problem, "First")
def regretk_inserts(removed_solution, removed_calls, k, problem, insert_type="First"):
	cargo_dict = problem['cargo_dict']
	best_posistions = {call: [] for call in removed_calls}
	for call in removed_calls:
		for vessel_idx in cargo_dict[call]:
			regret_values = regretk_value_one_vessel(removed_solution[vessel_idx], vessel_idx, call, problem)
			if regret_values:
				best_posistions[call].extend(regret_values)
	
	regret_values = {}
	inserts = {}
	for call, values in best_posistions.items():
		if values:
			values.sort(key=lambda x: x[0])
			if len(values) < k:
				regret_values[call] = float('inf')
			else:
				regret_values[call] = values[k-1][0] - values[0][0]
			inserts[call] = values[0]
	ordering = sorted(regret_values, key=regret_values.get, reverse=True)

	# probabilities = problem['probabilities'][:len(ordering)]
	# ws = [w/sum(probabilities) for w in probabilities]
	# new_prob = {}
	# for i, call in enumerate(ordering):
	# 	new_prob[call] = ws[i]
	# ordering = [call for call, _ in sorted(new_prob.items(), key=lambda x: random() * x[1], reverse=True)]

	for call in ordering:
		cost, i, j, vessel_idx = inserts[call]
		success, new_vessel = insert_op(vessel_idx, removed_solution[vessel_idx], call, problem, insert_type)
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
	return removed_solution
def regretk_value_one_vessel(vessel, vessel_idx, call, problem):
	# print("Regret value")
	regret_values = []
	for i in range(len(vessel) + 1):
		for j in range(i, len(vessel) + 2):
			new_list = deepcopy(vessel)
			new_list.insert(i, call)
			new_list.insert(j, call)
			f, c = feasability_one_vessel(new_list, problem, vessel_idx, spesific_cargo=call)
			if f:
				new_cost = travel_cost_one_vessel(new_list, problem, vessel_idx)
				regret_values.append((new_cost, i, j, vessel_idx))
	# for i in range(len(vessel) + 1):
	# 	pickup_temp = deepcopy(vessel)
	# 	pickup_temp.insert(i, call)
	# 	f1, c1 = feasability_one_vessel(pickup_temp, problem, vessel_idx, spesific_cargo=call)
	# 	if f1 or c1 == "Vessel capacity exceeded":
	# 		for j in range(i + 1, len(vessel) + 2):
	# 			deliver_temp = deepcopy(pickup_temp)
	# 			deliver_temp.insert(j, call)
	# 			f2, c2 = feasability_one_vessel(deliver_temp, problem, vessel_idx, spesific_cargo=call)
	# 			if f2:
	# 				new_cost = cost_one_vessel(deliver_temp, problem, vessel_idx)
	# 				regret_values.append((new_cost, i, j, vessel_idx))
	# 			elif c2 == "Time window wrong for given cargo":
	# 				break
	# 	elif c1 == "Time window wrong for given cargo":
	# 		break
	# regret_values.sort(key=lambda x: x[0])
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
			success, temp_vessel = insert_op(vessel_idx, vessel, call, problem, insert_type)
			if temp_vessel != vessel and success:
				new_solution = deepcopy(this_call_solution)
				new_solution[vessel_idx] = temp_vessel
				# this_call_solution = deepcopy(new_solution)
				# break
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
	return ret_solution
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
	outer_range = list(range(len(vessel) + 1))
	shuffle(outer_range)
	for i in outer_range:
		inner_range = list(range(i, len(vessel) + 2))
		shuffle(inner_range)
		for j in inner_range:
			new_list = vessel.copy()
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
			new_list = vessel.copy()
			new_list.insert(i, call)
			new_list.insert(j, call)
			f, c = feasability_one_vessel(new_list, problem, vessel_idx, spesific_cargo=call)
			if f:
				new_cost = cost_one_vessel(new_list, problem, vessel_idx)
				if new_cost < best_cost and (random() < 0.8 or best_cost == float('inf')):
					best_cost = new_cost
					best_list = deepcopy(new_list)
	if best_cost == float('inf'):
		return False, vessel
	return True, best_list
def insert_call_random(vessel_idx, vessel, call, problem):
	if (tuple(vessel), call) in memo and vessel:
		return memo[(tuple(vessel), call)]
	poss = []
	for i in range(len(vessel) + 1):
		for j in range(i, len(vessel) + 2):
			new_list = deepcopy(vessel)
			new_list.insert(i, call)
			new_list.insert(j, call)
			f, c = feasability_one_vessel(new_list, problem, vessel_idx, spesific_cargo=call)
			if f:
				poss.append(new_list)
	if not poss:
		memo[(tuple(vessel), call)] = False, vessel
		return False, vessel
	memo[(tuple(vessel), call)] = True, sample(poss, 1)[0]
	return True, sample(poss, 1)[0]
def remove_random_calls(solution, problem, n):
		n_calls = problem['n_calls']
		calls = sample(range(1, n_calls + 1), n)
		new_solution = [[c for c in v if c not in calls] for v in solution]
		return new_solution, calls
def remove_expensive_travel_calls(solution, problem, n):
	probabilities = problem['probabilities']
	n_calls = problem['n_calls']
	costs = set()
	for v_idx, temp_vessel in enumerate(solution[:-1]):
		if len(temp_vessel) == 0:
			continue
		pre_cost = travel_cost_one_vessel(temp_vessel, problem, v_idx)
		for cargo in set(temp_vessel):
			temp_vessel = [c for c in temp_vessel if c != cargo]
			post_cost = travel_cost_one_vessel(temp_vessel, problem, v_idx)
			costs.add((cargo, pre_cost - post_cost, v_idx))
	costs = list(costs)
	if len(costs) < n:
		return remove_outsourced_calls(solution, problem, n)
	amount = min(n, len(costs))
	costs.sort(key=lambda x: x[1], reverse=True)
	ws = [w / sum(probabilities) for w in probabilities]

	# calls = weighted_sample_without_replacement([x[0] for x in costs], ws, amount)
 
	calls = weighted_sample_without_replacement([x[0] for x in costs], ws, amount // 2)
	calls.extend(sample([x for x in range(1, problem['n_calls'] + 1) if x not in calls], amount - (amount // 2)))
 
	# calls = [x[0] for x in costs[:amount]]
	# calls.extend(sample([x for x in range(1, problem['n_calls'] + 1) if x not in calls], amount - len(calls)))

	new_solution = [[c for c in v if c not in calls] for v in solution]
	
	return new_solution, calls
def remove_expensive_calls(solution, problem, n):
	# TODO, take half most expesive, and rest random
	Cargo = problem['Cargo']
	probabilities = problem['probabilities']
	costs = set()
	for v_idx, temp_vessel in enumerate(solution[:-1]):
		if len(temp_vessel) == 0:
			continue
		pre_cost = cost_one_vessel(temp_vessel, problem, v_idx)
		for cargo in set(temp_vessel):
			temp_vessel = [c for c in temp_vessel if c != cargo]
			post_cost = cost_one_vessel(temp_vessel, problem, v_idx)
			costs.add((cargo, pre_cost - post_cost, v_idx))
	# costs.update([(j, Cargo[j - 1, 3] / 2, -1) for j in solution[-1]])
		
	costs = list(costs)
	if len(costs) == 0:
		return remove_outsourced_calls(solution, problem, n)
	amount = min(n, len(costs))
	costs.sort(key=lambda x: x[1], reverse=True)
	ws = [w/sum(probabilities) for w in probabilities]

	# calls = weighted_sample_without_replacement([x[0] for x in costs], ws, amount)

	calls = weighted_sample_without_replacement([x[0] for x in costs], ws, amount // 2)
	calls.extend(sample([x for x in range(1, problem['n_calls'] + 1) if x not in calls], amount - (amount // 2)))
	
	# calls = [x[0] for x in costs[:amount // 2]]
	# calls.extend(sample([x for x in range(1, problem['n_calls'] + 1) if x not in calls], amount - len(calls)))

	new_solution = [[c for c in v if c not in calls] for v in solution]
	return new_solution, calls
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
	return new_solution, calls
def remove_similar_calls(solution, problem, n):
	n_calls = problem['n_calls']
	probabilities = problem['probabilities']
	similarity_scores = problem['similarity_scores']
	random_call = randint(1, n_calls)
	similarities = similarity_scores[random_call - 1]
	poss = [(i + 1, similarities[i]) for i in range(n_calls) if i + 1 != random_call]
	poss.sort(key=lambda x: x[1])
	ws = [w / sum(probabilities) for w in probabilities]
	calls = weighted_sample_without_replacement([x[0] for x in poss], ws, n)
	# calls = [random_call] + [x[0] for x in poss[:n]]
	new_solution = [[c for c in v if c not in calls] for v in solution]
	return new_solution, calls

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
		'similarity_scores': prob['similarity_scores'],
		'cargo_dict': cargo_dict,
		'probabilities': probabilities,
		'feasabilityDict': {},
		'oneVesselCosts': {},
		'totalCosts': {},
		'opDeltas': {i:[] for i in range(5)},
	}

	for test in range(num_tests):

		solution = solution_to_list(init_solution(n_vehicles, n_calls), n_vehicles)

		start_time = time()

		cur_best_list, found_best_at = optimize_solution(solution, problem, iterations, print_iter)
		cur_best = list_to_solution(cur_best_list)
		solutions.append(cur_best)
		avg_times.append(time() - start_time)

		if cost_function(cur_best, problem) < cost_function(best_solution, problem):
			best_solution = cur_best

		print(f"Done with test {test + 1}, Took {round(avg_times[test], 2)}s")
		print(f"cost: {cost_function(cur_best, problem):,} Found on iteration: {found_best_at}")
		print(f"improvement: {round(100 * (cost_function(init_solution(n_vehicles, n_calls), problem) - cost_function(cur_best, problem)) / cost_function(init_solution(n_vehicles, n_calls), problem), 2)}%")
		print(f"Sol: {cur_best}")
		print()

	return avg_times, solutions, best_solution, problem

def main():
	num_tests = 1
	prob_load = 3
	total_iterations = 10000
	should_print_sol = True
	should_print_iter = False
	problems = ['Call_7_Vehicle_3.txt', 'Call_7_Vehicle_3.txt', 'Call_18_Vehicle_5.txt',
				'Call_35_Vehicle_7.txt', 'Call_80_Vehicle_20.txt', 'Call_130_Vehicle_40.txt',
				'Call_300_Vehicle_90.txt']
	res = []
	if prob_load == 0:
		for i in range(1,len(problems)):
			print("Problem: ", problems[i], "test number: ", i)
			prob = init_problem(problems, i)
			avg_times, solutions, best_solution, problem = run_tests(num_tests, total_iterations, prob, print_iter=False)
			res.append((avg_times, solutions, problem, num_tests, problems, i, best_solution, should_print_sol, "Simulated annealing 4ops"))

		for r in res:
			pprint(*r)
	else:
		prob = init_problem(problems, prob_load)
		avg_times, solutions, best_solution = run_tests(num_tests, total_iterations, prob, print_iter=should_print_iter)
		pprint(avg_times, solutions, problem, num_tests, problems, prob_load, best_solution, solve_method="Simulated annealing", print_best=should_print_sol)

if __name__ == '__main__':
	cProfile.run('main()', "t.prof")
	# main()
