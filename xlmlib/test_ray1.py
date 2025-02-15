import ray

# Initialize Ray
ray.init()


# Define a remote function
@ray.remote 
def square(x):
    return x * x

# Create a list of tasks
numbers = [1, 2, 3, 4, 5]
futures = [square.remote(num) for num in numbers]

# Retrieve results
results = ray.get(futures)

print("Squares:", results)

ray.shutdown()