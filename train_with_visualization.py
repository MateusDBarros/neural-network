"""
Neural Network Training with Real-Time Visualization

This script trains the custom neural network from NeuralNetwork.py
with live visualization of training metrics, weights, and activations.
"""

import numpy as np
from NeuralNetwork import (
    Network, Adam, 
    categorical_cross_entropy, categorical_cross_entropy_grad,
    X_train, Y_train, X_test, Y_test,
    get_batches, accuracy
)
from visualizer import NeuralNetworkVisualizer


def train_with_visualization(
    layer_config=[784, 128, 'batchnorm', 'dropout', 64, 'batchnorm', 'dropout', 10],
    learning_rate=0.001,
    epochs=20,
    batch_size=64,
    visualization_interval=1
):
    """
    Train neural network with real-time visualization
    
    Args:
        layer_config: Network architecture configuration
        learning_rate: Learning rate for Adam optimizer
        epochs: Number of training epochs
        batch_size: Batch size for training
        visualization_interval: Update visualization every N epochs
    """
    
    print("=" * 70)
    print("Neural Network Training with Real-Time Visualization")
    print("=" * 70)
    print(f"\nNetwork Architecture: {layer_config}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"\nTraining on {len(X_train)} samples")
    print(f"Testing on {len(X_test)} samples")
    print("\nStarting training...\n")
    
    # Initialize network and optimizer
    net = Network(layer_config, output='softmax')
    optimizer = Adam(learning_rate=learning_rate)
    
    # Initialize visualizer
    viz = NeuralNetworkVisualizer(max_history=epochs)
    
    # Get a sample for activation visualization
    sample_input = X_test[:1]
    
    try:
        for epoch in range(epochs):
            # Training phase
            net.train()
            total_loss = 0
            batches = 0
            
            for X_batch, Y_batch in get_batches(X_train, Y_train, batch_size):
                predicted = net.forward(X_batch)
                total_loss += categorical_cross_entropy(predicted, Y_batch)
                grad = categorical_cross_entropy_grad(predicted, Y_batch)
                net.backward(grad, optimizer)
                batches += 1
            
            # Evaluation phase
            net.eval()
            test_pred = net.forward(X_test)
            test_acc = accuracy(test_pred, Y_test)
            avg_loss = total_loss / batches
            
            # Print progress
            print(f"Epoch {epoch+1:2d}/{epochs} — Loss: {avg_loss:.4f}  Test Accuracy: {test_acc*100:.1f}%")
            
            # Update visualization
            if (epoch + 1) % visualization_interval == 0 or epoch == 0:
                viz.update_metrics(epoch + 1, avg_loss, test_acc)
                viz.update_weights(net)
                viz.update_activations(net, sample_input)
                viz.plot()
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
        print(f"Final Loss: {avg_loss:.4f}")
        
        # Save final visualization
        viz.save_figure('final_training_visualization.png')
        
        # Keep window open
        print("\nVisualization window is open. Close it to exit.")
        input("Press Enter to close and exit...")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        viz.save_figure('interrupted_training_visualization.png')
    
    finally:
        viz.close()
    
    return net, viz


if __name__ == '__main__':
    # Train with default configuration
    trained_network, visualizer = train_with_visualization(
        layer_config=[784, 128, 'batchnorm', 'dropout', 64, 'batchnorm', 'dropout', 10],
        learning_rate=0.001,
        epochs=20,
        batch_size=64,
        visualization_interval=1  # Update every epoch
    )
    
    print("\nTraining session completed successfully!")

# Made with Bob
