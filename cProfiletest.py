import cProfile
import pstats

# Run your script with cProfile
cProfile.run('run_tests()', 'test.prof')

# Analyze the profile data
stats = pstats.Stats('test.prof')
stats.sort_stats(pstats.SortKey.TIME)  # You can choose a sorting method
stats.print_stats()