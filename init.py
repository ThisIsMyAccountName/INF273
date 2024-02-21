from pdp_utils.pdp_utils.Utils import load_problem, feasibility_check, cost_function

def init_problem(problems, prob_load):
	problem_file = f'problems/{problems[int(prob_load)]}'
	return load_problem(problem_file)

def init_solution(n_vehicles, n_calls):
	sol = [0] * n_vehicles + list(i for i in range(1, n_calls + 1) for _ in range(2))
	return sol

def pprint(avg_times, solutions, prob, num_tests, problems, prob_load, best_solution, print_best=True, solve_method="Random"):
	avg_time = sum(avg_times) / num_tests if num_tests else 0
	avg_cost = sum(cost_function(sol, prob) for sol in solutions) / num_tests if num_tests else 0
	best_cost = min(cost_function(sol, prob) for sol in solutions) if num_tests else 0
	init_cost = cost_function(init_solution(prob['n_vehicles'], prob['n_calls']), prob)
	improvement = 100 * (init_cost - best_cost) / init_cost if init_cost else 0

	# print()
	# for sol in solutions:
	# 	print(sol, cost_function(sol, prob), feasibility_check(sol, prob))
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
	print(f"|{solve_method}|{round(avg_cost, 2)}|{round(best_cost, 2)}|{round(improvement, 2)}%|{round(avg_time, 2)}s|")
	print()
	if print_best:
		print(f"Initial cost: {init_cost}")
		print(f"Best cost: {best_cost}")
		print(f"Best solution: {best_solution}")
	