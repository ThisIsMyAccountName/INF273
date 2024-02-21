
from time import time
from random import shuffle, randint
from pdp_utils.pdp_utils.Utils import load_problem, feasibility_check, cost_function

tests = 10
prob_load = 1
problems = ['Call_7_Vehicle_3.txt', 'Call_7_Vehicle_3.txt', 'Call_18_Vehicle_5.txt', 'Call_35_Vehicle_7.txt', 'Call_80_Vehicle_20.txt', 'Call_130_Vehicle_40.txt', 'Call_300_Vehicle_90.txt']
prob = load_problem('problems/' + problems[int(prob_load)])


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
print(VesselCargo)
iterations = 10000
# makes a list of the form [1, 1, 2, 2, 3, 3, ..., n, n]
expand = lambda n: [i for i in range(1, n+1) for _ in range(2)]
lst = [0] * n_vehicles + list(range(1, n_calls + 1)) + list(range(1, n_calls + 1))
avg_times = []
solutions = []
lowest_cost = float('inf')
best = [0] * n_vehicles + expand(n_calls)
for test in range(tests):

	start_time = time()

	cur_best = [0] * n_vehicles + expand(n_calls)
	lowest_cost = cost_function(cur_best, prob)
	t = [0] * n_vehicles + list(range(1, n_calls + 1))
	next_loop = 0
	for iter in range(iterations):
		shuffle(t)
		sol = []
		i = 0
		for _ in range(n_vehicles):
			temp = [] 
			while t[i]:
				temp.append(t[i])
				temp.append(t[i])
				i += 1
			i += 1
			shuffle(temp)
			sol.extend(temp)
			sol.append(0)
		sol.extend(t[i:])
		sol.extend(t[i:])

		feasible, c = feasibility_check(sol, prob)
		if not feasible:
			continue
		if (cost:= cost_function(sol, prob)) < lowest_cost:
			cur_best = sol
			lowest_cost = cost
			# print("new lowest cost: ", lowest_cost, "for test number: ", test, "iteration: ", iter, "sol: ", sol)
		
	solutions.append(cur_best)
	avg_times.append(time() - start_time)
	if cost_function(cur_best, prob) < cost_function(best, prob):
		best = cur_best
	
	print("Done with test number:", test+1, "It took", round(avg_times[test], 2), "seconds")

avg_time = sum(avg_times)/tests if tests else 0
avg_cost = sum(cost_function(sol, prob) for sol in solutions)/tests if tests else 0
best_cost = min(cost_function(sol, prob) for sol in solutions) if tests else 0
initial_cost = cost_function([0] * n_vehicles + expand(n_calls), prob)
improvement = 100 * (initial_cost - best_cost)/initial_cost if initial_cost else 0

print(f"Average cost: {avg_cost}")
print(f"Best cost: {best_cost}")
print(f"Improvement: {improvement}%")
print(f"Running time: {avg_time}")
print()
print(f"|{problems[prob_load][:-4]}|||||")
print("|:-:|:-:|:-:|:-:|:-:|")
print("|✅|Average objective|Best objective|Improvement (%)|Running time|")
print(f"|Random|{round(avg_cost, 2)}|{round(best_cost, 2)}|{round(improvement, 2)}%|{round(avg_time, 2)}s|")
print()
print()
print("Best solution:", best)