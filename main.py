import argparse
from src import mnist_loader 

def main(dataset_path):
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(dataset_path)

    from src import network2
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, lmbda=0.1, 
            monitor_evaluation_accuracy=True, monitor_evaluation_cost=True,
            monitor_training_accuracy=True, monitor_training_cost=True)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train a neural network on the MNIST dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the MNIST dataset file (e.g., data/mnist.pkl.gz)")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the dataset path
    main(args.dataset_path)