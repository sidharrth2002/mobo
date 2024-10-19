import tensorflow as tf
import numpy as np
import os
import json

# Ensure TensorFlow uses the proper log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set up the MultiWorkerMirroredStrategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# MapReduce-like task using TensorFlow 2.x
def map_reduce_task():
    # Sample data: let's say we want to sum numbers using two workers
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    
    # Split the data between workers
    num_workers = strategy.num_replicas_in_sync
    split_data = np.array_split(data, num_workers)
    
    tf_config = json.loads(os.environ['TF_CONFIG'])
    task_index = tf_config['task']['index']

    # Use a dataset to distribute data processing
    dataset = tf.data.Dataset.from_tensor_slices(split_data[task_index]).batch(2)

    # Define a simple model to simulate the reduction operation
    def create_model():
        # For simplicity, this model will just sum the inputs
        inputs = tf.keras.layers.Input(shape=(1,))
        outputs = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x))(inputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='sgd', loss='mse')
        return model

    # Use the strategy scope to create the model
    with strategy.scope():
        model = create_model()

    # Training the model to perform the sum (map-reduce)
    print(dataset)
    model.fit(dataset, epochs=1)
    
    # Perform the reduction (sum)
    sum_result = tf.reduce_sum(model.predict(dataset))
    print(f"Worker {os.environ['TF_CONFIG'][-2]} - Sum Result: {sum_result.numpy()}")

if __name__ == "__main__":
    map_reduce_task()