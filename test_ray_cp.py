import ray
import time

# Initialize Ray
ray.init()

# Define the producer function
@ray.remote
def producer(num_tasks):
    for i in range(num_tasks):
        task = i
        print(f"Produced task {task}")
        yield task
        time.sleep(0.5)

# Define the consumer function to run on GPU
@ray.remote(num_gpus=1)
def consumer(task):
    gpu_id = ray.get_gpu_ids()  # Get the allocated GPU ID
    print(f"Processing task {task} on GPU {gpu_id}")
    time.sleep(1)  # Simulate some processing time
    result = f"Task {task} processed on GPU {gpu_id}"
    return result

# Main producer-consumer function
def run_producer_consumer(num_tasks, num_consumers):
    # Start producer

    producer_iterator = producer.remote(num_tasks)
    #print("Producing tasks...")
    #tasks = ray.get(producer.remote(num_tasks))

    # Send tasks to consumers for processing
    results = []
    for task in producer_iterator:
        result = consumer.remote(task)  # Send task to available consumers
        results.append(result)

    # Gather and print results
    for result in ray.get(results):
        print(result)

# Run the producer-consumer system
if __name__ == "__main__":
    num_tasks = 1000
    num_consumers = 4  # You can adjust this based on available GPUs
    run_producer_consumer(num_tasks, num_consumers)

    # Shut down Ray
    ray.shutdown()