import numpy as np
from copy import deepcopy
from random import sample, random
from enum import Enum
import numpy as np

minus_one = lambda sublist: list(map(lambda x: x-1, sublist))

def solution_to_list(solution, vecs):
	ret_sol = []
	i = 0
	for _ in range(vecs):
		new_vec = []
		while solution[i] != 0:
			new_vec.append(solution[i])
			i += 1
		i += 1 
		ret_sol.append(new_vec)
	ret_sol.append(solution[i:])
	return ret_sol

def cargo_to_vessel(bitmask, n_calls):
	ret = {i+1:[] for i in range(n_calls)}
	for i,vec in enumerate(bitmask):
		for j,cargo in enumerate(vec):
			if cargo:
				ret[j+1].append(i)
	return ret

def list_to_solution(lists, delimiter=0):
	result = [i for sublist in zip(lists, [[delimiter]] * (len(lists))) for item in sublist for i in item]
	return result[:-1]

def remove_elem_from(list_solution, cargo_to_move, move_from):
	list_solution[move_from].remove(cargo_to_move)
	list_solution[move_from].remove(cargo_to_move)

def add_elem_to(list_solution, cargo_to_move, move_to):
	list_solution[move_to].append(cargo_to_move)
	list_solution[move_to].append(cargo_to_move)

	
def insert_number_at_all_positions_pick_first(original_list, to_vessel, from_vessel, number_to_insert, prob, memo):
	sol_copy = deepcopy(original_list)
	for i in range(len(original_list[to_vessel]) + 1):
		for j in range(i + 1, len(original_list[to_vessel]) + 2):
			new_list = deepcopy(original_list[to_vessel])
			new_list.insert(i, number_to_insert)
			new_list.insert(j, number_to_insert)
			if tuple(new_list) in memo:
				continue
			f, _ = feasability_one_vessel(new_list, prob, to_vessel)
			if f:
				sol_copy[to_vessel] = new_list
				return True, sol_copy
			memo.add(tuple(new_list))
	
	add_elem_to(original_list, number_to_insert, from_vessel)
	return False, original_list

def insert_number_at_all_positions_pick_random(original_list, to_vessel, from_vessel, number_to_insert, prob, memo):
	sol_copy = deepcopy(original_list)
	for i in range(len(original_list[to_vessel]) + 1):
		for j in range(i + 1, len(original_list[to_vessel]) + 2):
			new_list = deepcopy(original_list[to_vessel])
			new_list.insert(i, number_to_insert)
			new_list.insert(j, number_to_insert)
			if tuple(new_list) in memo:
				continue
			sol_copy[to_vessel] = new_list
			if feasability_one_vessel(new_list, prob, to_vessel)[0] and random() < (1 / (len(new_list) / 3)):
				sol_copy[to_vessel] = new_list
				return True, sol_copy
			memo.add(tuple(new_list))
	add_elem_to(original_list, number_to_insert, from_vessel)
	return False, original_list

def insert_number_at_all_positions_pick_best(original_list, to_vessel, from_vessel, number_to_insert, prob, memo):
	l = []
	sol_copy = deepcopy(original_list)
	for i in range(len(original_list[to_vessel]) + 1):
		for j in range(i + 1, len(original_list[to_vessel]) + 2):
			new_list = deepcopy(original_list[to_vessel])
			new_list.insert(i, number_to_insert)
			new_list.insert(j, number_to_insert)
			sol_copy[to_vessel] = new_list
			if feasability_one_vessel(new_list, prob, to_vessel)[0]:
				l.append(new_list)
	if not l:
		add_elem_to(original_list, number_to_insert, from_vessel)
		return False, original_list
	best = min(l, key=lambda x: cost_one_vessel(x, prob, to_vessel))
	sol_copy[to_vessel] = best
	return True, sol_copy

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
	# while most_expensive and (most_expensive[-1][1], most_expensive[-1][2]) in memo:
	# 	most_expensive.pop()
	# if not most_expensive:
	# 	return pick_random_cargo(list_solution)
	# memo.add((most_expensive[-1][1], most_expensive[-1][2]))
	return most_expensive[-1][1], most_expensive[-1][2], most_expensive

def Move_x_to_outsource(list_solution, x, prob):
	n_vehicles = prob['n_vehicles']
	n_calls = prob['n_calls']
	copy = deepcopy(list_solution)
	cargo_to_sample = [i for i in range(1, n_calls) if i not in copy[-1]]
	if len(cargo_to_sample) < x:
		return list_solution
	move_to_outsource = sample(cargo_to_sample, x)
	for cargo in move_to_outsource:
		for i in range(n_vehicles):
			if cargo in copy[i]:
				remove_elem_from(copy, cargo, i)
				break
	for cargo in move_to_outsource:
		add_elem_to(copy, cargo, -1)

	return copy


def move_x_most_expensive_to_outsource(list_solution, x, prob):
	lists = deepcopy(list_solution[:-1])
	n_vehicles = prob['n_vehicles']
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
	
	x_most_expensive = sorted(costs, key=lambda x: x[0])[-x:]

	for cargo in x_most_expensive:
		for i in range(n_vehicles):
			if cargo[1] in list_solution[i]:
				remove_elem_from(list_solution, cargo[1], cargo[2])
				break
		add_elem_to(list_solution, cargo[1], -1)

	return list_solution

def cost_one_vessel(sublist, prob, vessel_num, is_outsource=False):
	Cargo =	prob['Cargo']
	TravelCost = prob['TravelCost']
	FirstTravelCost = prob['FirstTravelCost']
	PortCost = prob['PortCost']
	costs = prob['oneVesselCosts']
	tuple_sublist = tuple(sublist)
	if tuple_sublist in costs:
		return costs[tuple_sublist]
	if not sublist:
		return 0
	cur_vehicle = np.array(list(map(lambda x: x-1, sublist)))
	if is_outsource:
		outsource_cost = np.sum(Cargo[cur_vehicle, 3]) / 2
		return outsource_cost
	if len(cur_vehicle) > 0:
		sortRout = np.sort(cur_vehicle, kind='mergesort')
		I = np.argsort(cur_vehicle, kind='mergesort')
		Indx = np.argsort(I, kind='mergesort')

		PortIndex = Cargo[sortRout, 0].astype(int)
		PortIndex[::2] = Cargo[sortRout[::2], 0]
		PortIndex = PortIndex[Indx] - 1

		Diag = TravelCost[vessel_num, PortIndex[:-1], PortIndex[1:]]

		FirstVisitCost = FirstTravelCost[vessel_num, int(Cargo[cur_vehicle[0], 0] - 1)]
		RouteTravelCost = np.sum(np.hstack((FirstVisitCost, Diag.flatten())))
		CostInPorts = np.sum(PortCost[vessel_num, cur_vehicle]) / 2

		costs[tuple_sublist] = RouteTravelCost + CostInPorts
		return RouteTravelCost + CostInPorts
	
def travel_cost_one_vessel(sublist, prob, vessel_num, is_outsource=False):
	Cargo =	prob['Cargo']
	TravelCost = prob['TravelCost']
	FirstTravelCost = prob['FirstTravelCost']
	PortCost = prob['PortCost']
	costs = prob['oneVesselCosts']
	if (tuple(sublist), vessel_num) in costs:
		return costs[(tuple(sublist), vessel_num)]
	if not sublist:
		return 0
	cur_vehicle = np.array(list(map(lambda x: x-1, sublist)))
	if is_outsource:
		outsource_cost = np.sum(Cargo[cur_vehicle, 3]) / 2
		costs[(tuple(sublist), vessel_num)] = outsource_cost
		return outsource_cost
	if len(cur_vehicle) > 0:
		sortRout = np.sort(cur_vehicle, kind='mergesort')
		I = np.argsort(cur_vehicle, kind='mergesort')
		Indx = np.argsort(I, kind='mergesort')

		PortIndex = Cargo[sortRout, 0].astype(int)
		PortIndex[::2] = Cargo[sortRout[::2], 0]
		PortIndex = PortIndex[Indx] - 1

		Diag = TravelCost[vessel_num, PortIndex[:-1], PortIndex[1:]]

		FirstVisitCost = FirstTravelCost[vessel_num, int(Cargo[cur_vehicle[0], 0] - 1)]
		RouteTravelCost = np.sum(np.hstack((FirstVisitCost, Diag.flatten())))

		costs[(tuple(sublist), vessel_num)] = RouteTravelCost
		return RouteTravelCost
	
def feasability_one_vessel(sublist, prob, vessel_num, spesific_cargo=None):
	Cargo = prob['Cargo']
	TravelTime = prob['TravelTime']
	FirstTravelTime = prob['FirstTravelTime']
	VesselCapacity = prob['VesselCapacity']
	LoadingTime = prob['LoadingTime']
	UnloadingTime = prob['UnloadingTime']
	VesselCargo = prob['VesselCargo']
	feasabilityDict = prob['feasabilityDict']

	feasability = True
	c = ""
	hashable_solution = tuple(sublist)
	if (hashable_solution, vessel_num) in feasabilityDict:
		return feasabilityDict[(hashable_solution, vessel_num)]
	
	currentVPlan = minus_one(sublist)

	NoDoubleCallOnVehicle = len(currentVPlan)

	if NoDoubleCallOnVehicle > 0:
		
		if not np.all(VesselCargo[vessel_num, currentVPlan]):
			feasability = False
			c = "Cargo not compatible with vessel"
			return feasability, c
		Load_size = 0
		currentTime = 0
		sortedRout = np.sort(currentVPlan, kind='mergesort')
		I = np.argsort(currentVPlan, kind='mergesort')
		Indx = np.argsort(I, kind='mergesort')
		Load_size -= Cargo[sortedRout, 2]
		Load_size[::2] = Cargo[sortedRout[::2], 2]
		Load_size = Load_size[Indx]
		if np.any(VesselCapacity[vessel_num] - np.cumsum(Load_size) < 0):
			feasability = False
			c = "Vessel capacity exceeded"
			feasabilityDict[(hashable_solution, vessel_num)] = feasability, c
			return feasability, c
		Timewindows = np.zeros((2, NoDoubleCallOnVehicle))
		Timewindows[0] = Cargo[sortedRout, 6]
		Timewindows[0, ::2] = Cargo[sortedRout[::2], 4]
		Timewindows[1] = Cargo[sortedRout, 7]
		Timewindows[1, ::2] = Cargo[sortedRout[::2], 5]

		Timewindows = Timewindows[:, Indx]

		PortIndex = Cargo[sortedRout, 1].astype(int)
		PortIndex[::2] = Cargo[sortedRout[::2], 0]
		PortIndex = PortIndex[Indx] - 1

		LU_Time = UnloadingTime[vessel_num, sortedRout]
		LU_Time[::2] = LoadingTime[vessel_num, sortedRout[::2]]
		LU_Time = LU_Time[Indx]
		Diag = TravelTime[vessel_num, PortIndex[:-1], PortIndex[1:]]
		FirstVisitTime = FirstTravelTime[vessel_num, int(Cargo[currentVPlan[0], 0] - 1)]

		routeTravelTime = np.hstack((FirstVisitTime, Diag.flatten()))

		ArriveTime = np.zeros(NoDoubleCallOnVehicle)
		for j in range(NoDoubleCallOnVehicle):
			ArriveTime[j] = np.max((currentTime + routeTravelTime[j], Timewindows[0, j]))
			if ArriveTime[j] > Timewindows[1, j]:
				feasability = False
				c = "Time window wrong"
				
				if spesific_cargo == currentVPlan[j]:
					c = "Time window wrong for given cargo"
				feasabilityDict[(hashable_solution, vessel_num)] = feasability, c
				return feasability, c

			currentTime = ArriveTime[j] + LU_Time[j]
		feasabilityDict[(hashable_solution, vessel_num)] = feasability, c
	return feasability, c