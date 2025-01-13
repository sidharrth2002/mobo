import ray

ray.init(address="auto")

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get_counter(self):
        return self.value

# Create an actor from this class.
counter = Counter.remote()

counter_ids = [counter.increment.remote() for _ in range(4)]

# Get the results of the tasks as they complete.
results = ray.get(counter_ids)

print(results)  # [1, 2, 3, 4]