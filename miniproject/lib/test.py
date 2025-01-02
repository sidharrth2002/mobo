import math

# front = [4]
# crowding_distance = [math.inf]

# sorted_front = sorted(front, key=lambda x: -crowding_distance[x])

reduction_factor = 3
test = [1, 2, 3]

test = test[:max(1, math.ceil(1/(1-3)))]
print(test)