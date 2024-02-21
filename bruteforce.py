def insert_number_at_all_positions(original_list, number_to_insert):
    result_list = []
    
    for i in range(len(original_list) + 1):  # Range adjusted for the first insertion
        for j in range(i + 1, len(original_list) + 2):  # Range adjusted for the second insertion
            new_list = original_list.copy()
            new_list.insert(i, number_to_insert)
            new_list.insert(j, number_to_insert)
            result_list.append(new_list)
    
    return result_list

# Example usage:
original_list = [1, 2, 3, 4]
number_to_insert = 99

result = insert_number_at_all_positions(original_list, number_to_insert)
print(result)


def swap_inside_vessel(list_solution, prob, vessel):
    cur_cost = cost_function(list_to_solution(list_solution), prob)
    working = list_solution[vessel]
    n_calls = len(working)
    cur_best_swap = (0, 0, 0, 0)
    # print("working: ", working)
    for i in range(n_calls):
        for j in range(i+1, n_calls):
            working[i], working[j] = working[j], working[i]
            # print("swapping: ", working[i], working[j],i,j, working)
            # print("old cost: ", cur_cost)
            # print("new_cost:" , cost_function(list_to_solution(list_solution), prob))
            # print()
            if cur_cost > cost_function(list_to_solution(list_solution), prob):
                print("swapping: ", working[i], working[j],i,j, working)
                print("old cost: ", cur_cost)
                print("new_cost:" , cost_function(list_to_solution(list_solution), prob))
                print()
                cur_cost = cost_function(list_to_solution(list_solution), prob)
                cur_best_swap = (i, j, working[i], working[j])

            working[i], working[j] = working[j], working[i]
    if cur_best_swap != (0, 0, 0, 0):
        working[cur_best_swap[0]], working[cur_best_swap[1]] = working[cur_best_swap[1]], working[cur_best_swap[0]]
        print("best swap: ", cur_best_swap)
        print("new cost: ", cur_cost)
        print("new working: ", working)
        print()



def swap_inside_vessel(list_solution, prob, vessel, recursion=0):
	cur_cost = cost_function(list_to_solution(list_solution), prob)
	working = list_solution[vessel]
	n_calls = len(working)
	# print("working: ", working)
	for i in range(n_calls):
		for j in range(i+1, n_calls):
			working[i], working[j] = working[j], working[i]
			# print("swapping: ", working[i], working[j],i,j, working)
			# print("old cost: ", cur_cost)
			# print("new_cost:" , cost_function(list_to_solution(list_solution), prob))
			# print()
			if feasibility_check(list_to_solution(list_solution), prob)[0]:
				if cur_cost > cost_function(list_to_solution(list_solution), prob):
					if recursion < 10:
						swap_inside_vessel(list_solution, prob, vessel, recursion+1)
					# print("swapping: ", working[i], working[j],i,j, working)
					# print("old cost: ", cur_cost)
					# print("new_cost:" , cost_function(list_to_solution(list_solution), prob))
					# print()
			working[i], working[j] = working[j], working[i]