from time import time
from copy import deepcopy
from random import sample, randint, shuffle
from pdp_utils.pdp_utils.Utils import load_problem, feasibility_check, cost_function

def init_problem(problems, prob_load):
	problem_file = f'problems/{problems[int(prob_load)]}'
	return load_problem(problem_file)

def init_solution(n_vehicles, n_calls):
	return [0] * n_vehicles + list(range(1, n_calls + 1)) + list(range(1, n_calls + 1))

def random_local_search(solution):
	v1, v2 = sample(range(len(solution)), 2)
	solution[v1], solution[v2] = solution[v2], solution[v1]

def random_feasable(n_vehicles, vessel_cargo):
	sol = []
	for vic in range(n_vehicles):
		possibilities = [i for i in range(len(vessel_cargo[0])) if vessel_cargo[vic][i] == 1 and i not in sol]
		choises = sample(possibilities, randint(0,len(possibilities))) * 2
		shuffle(choises)
		sol.extend(choises)
		sol.append(0)

	rest = [i for i in range(len(vessel_cargo[0])) if i not in sol] * 2
	sol.extend(rest)

	return sol

def optimize_solution(solution, prob, iterations):
	best_sol = deepcopy(solution)
	lowest_cost = cost_function(best_sol, prob)

	for _ in range(iterations):
		
		solution = random_feasable(prob['n_vehicles'], prob['VesselCargo'])

		feasible, _ = feasibility_check(solution, prob)
		if not feasible:
			continue

		if (x:= cost_function(solution ,prob)) < lowest_cost:
			best_sol = deepcopy(solution)
			lowest_cost = x

	return best_sol

def run_tests(num_tests, iterations, prob):
	avg_times, solutions = [], []
	best_solution = init_solution(prob['n_vehicles'], prob['n_calls'])

	for test in range(num_tests):
		start_time = time()

		init_sol = init_solution(prob['n_vehicles'], prob['n_calls'])
		cur_best = optimize_solution(init_sol, prob, iterations)

		solutions.append(cur_best)
		avg_times.append(time() - start_time)

		if cost_function(cur_best, prob) < cost_function(best_solution, prob):
			best_solution = cur_best

		print(f"Done with test {test + 1}, Took {round(avg_times[test], 2)}s, Cost: {cost_function(cur_best, prob)}")

	return avg_times, solutions, best_solution

def main():
	num_tests = 10
	prob_load = 1
	problems = ['Call_7_Vehicle_3.txt', 'Call_7_Vehicle_3.txt', 'Call_18_Vehicle_5.txt',
				'Call_35_Vehicle_7.txt', 'Call_80_Vehicle_20.txt', 'Call_130_Vehicle_40.txt',
				'Call_300_Vehicle_90.txt']
	prob = init_problem(problems, prob_load)
	iterations = 10000
	avg_times, solutions, best_solution = run_tests(num_tests, iterations, prob)

	avg_time = sum(avg_times) / num_tests if num_tests else 0
	avg_cost = sum(cost_function(sol, prob) for sol in solutions) / num_tests if num_tests else 0
	best_cost = min(cost_function(sol, prob) for sol in solutions) if num_tests else 0
	init_cost = cost_function(init_solution(prob['n_vehicles'], prob['n_calls']), prob)
	improvement = 100 * (init_cost - best_cost) / init_cost if init_cost else 0

	print()
	for sol in solutions:
		print(sol, cost_function(sol, prob), feasibility_check(sol, prob))
	print()
	print(f"Avg cost: {avg_cost}")
	print(f"Best cost: {best_cost}")
	print(f"Improvement: {improvement}%")
	print(f"Avg Run time: {avg_time}")
	print(f"Total Run time: {sum(avg_times)}")
	print()
	print(f"|{problems[prob_load][:-4]}|||||")
	print("|:-:|:-:|:-:|:-:|:-:|")
	print(f"|✅|Avg objective|Best objective|Improvement (%)|Run time|")
	print(f"|Random|{round(avg_cost, 2)}|{round(best_cost, 2)}|{round(improvement, 2)}%|{round(avg_time, 2)}s|")
	print()
	print(f"Initial cost: {init_cost}")
	print(f"Best cost: {best_cost}")
	print(f"Best solution: {best_solution}")

if __name__ == "__main__":
	main()
