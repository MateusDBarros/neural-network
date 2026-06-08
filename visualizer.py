import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from collections import deque


class NeuralNetworkVisualizer:
    """Real-time visualization for neural network training"""
    
    def __init__(self, max_history=100):
        """
        Initialize the visualizer with interactive plotting
        
        Args:
            max_history: Maximum number of epochs to keep in history
        """
        self.max_history = max_history
        
        # Training metrics history
        self.train_losses = deque(maxlen=max_history)
        self.test_accuracies = deque(maxlen=max_history)
        self.epochs = deque(maxlen=max_history)
        
        # Weight and activation data
        self.weight_data = []
        self.activation_data = []
        
        # Setup the figure with subplots
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.suptitle('Neural Network Training Visualization', fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Loss plot (top left, spans 2 columns)
        self.ax_loss = self.fig.add_subplot(gs[0, :2])
        self.ax_loss.set_title('Training Loss Over Time')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        self.line_loss, = self.ax_loss.plot([], [], 'b-', linewidth=2, label='Training Loss')
        self.ax_loss.legend()
        
        # Accuracy plot (top right)
        self.ax_acc = self.fig.add_subplot(gs[0, 2])
        self.ax_acc.set_title('Test Accuracy')
        self.ax_acc.set_xlabel('Epoch')
        self.ax_acc.set_ylabel('Accuracy (%)')
        self.ax_acc.grid(True, alpha=0.3)
        self.ax_acc.set_ylim(0, 100)
        self.line_acc, = self.ax_acc.plot([], [], 'g-', linewidth=2, label='Test Accuracy')
        self.ax_acc.legend()
        
        # Weight distribution (middle left)
        self.ax_weights = self.fig.add_subplot(gs[1, 0])
        self.ax_weights.set_title('Weight Distribution (Layer 1)')
        self.ax_weights.set_xlabel('Weight Value')
        self.ax_weights.set_ylabel('Frequency')
        
        # Gradient magnitude (middle center)
        self.ax_gradients = self.fig.add_subplot(gs[1, 1])
        self.ax_gradients.set_title('Gradient Magnitudes by Layer')
        self.ax_gradients.set_xlabel('Layer')
        self.ax_gradients.set_ylabel('Mean Gradient Magnitude')
        
        # Activation heatmap (middle right)
        self.ax_activations = self.fig.add_subplot(gs[1, 2])
        self.ax_activations.set_title('Layer Activations (Sample)')
        self.ax_activations.set_xlabel('Neuron Index')
        self.ax_activations.set_ylabel('Layer')
        
        # Network architecture visualization (bottom, spans all columns)
        self.ax_network = self.fig.add_subplot(gs[2, :])
        self.ax_network.set_title('Network Architecture & Activity')
        self.ax_network.axis('off')
        
        # Current stats text (bottom right corner)
        self.stats_text = self.fig.text(0.98, 0.02, '', 
                                        verticalalignment='bottom',
                                        horizontalalignment='right',
                                        fontsize=10,
                                        family='monospace',
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.show(block=False)
        
    def update_metrics(self, epoch, train_loss, test_accuracy):
        """
        Update training metrics
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            test_accuracy: Test accuracy (0-1 scale)
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.test_accuracies.append(test_accuracy * 100)  # Convert to percentage
        
    def update_weights(self, network):
        """
        Extract and store weight information from network
        
        Args:
            network: Neural network object with layers
        """
        self.weight_data = []
        for i, layer in enumerate(network.layers):
            if hasattr(layer, 'W'):
                self.weight_data.append({
                    'layer_id': i,
                    'weights': layer.W.flatten(),
                    'mean': np.mean(layer.W),
                    'std': np.std(layer.W)
                })
    
    def update_activations(self, network, sample_input):
        """
        Capture activations for a sample input
        
        Args:
            network: Neural network object
            sample_input: Input sample to forward through network
        """
        self.activation_data = []
        X = sample_input
        
        for i, layer in enumerate(network.layers):
            X = layer.forward(X)
            if hasattr(layer, 'W'):  # Only store for actual layers, not dropout
                self.activation_data.append({
                    'layer_id': i,
                    'activations': X[0] if len(X.shape) > 1 else X,  # First sample
                    'mean': np.mean(X),
                    'std': np.std(X)
                })
    
    def plot(self):
        """Update all plots with current data"""
        
        # Update loss plot
        if len(self.epochs) > 0:
            self.line_loss.set_data(list(self.epochs), list(self.train_losses))
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()
        
        # Update accuracy plot
        if len(self.epochs) > 0:
            self.line_acc.set_data(list(self.epochs), list(self.test_accuracies))
            self.ax_acc.relim()
            self.ax_acc.autoscale_view()
        
        # Update weight distribution
        if self.weight_data:
            self.ax_weights.clear()
            self.ax_weights.set_title('Weight Distribution (Layer 1)')
            self.ax_weights.set_xlabel('Weight Value')
            self.ax_weights.set_ylabel('Frequency')
            weights = self.weight_data[0]['weights']
            self.ax_weights.hist(weights, bins=50, alpha=0.7, color='blue', edgecolor='black')
            self.ax_weights.axvline(np.mean(weights), color='red', linestyle='--', 
                                   label=f'Mean: {np.mean(weights):.4f}')
            self.ax_weights.legend()
        
        # Update gradient magnitude plot
        if self.weight_data:
            self.ax_gradients.clear()
            self.ax_gradients.set_title('Weight Statistics by Layer')
            self.ax_gradients.set_xlabel('Layer')
            self.ax_gradients.set_ylabel('Weight Std Dev')
            layer_ids = [w['layer_id'] for w in self.weight_data]
            stds = [w['std'] for w in self.weight_data]
            self.ax_gradients.bar(layer_ids, stds, alpha=0.7, color='orange')
            self.ax_gradients.grid(True, alpha=0.3)
        
        # Update activation heatmap
        if self.activation_data:
            self.ax_activations.clear()
            self.ax_activations.set_title('Layer Activations (Sample)')
            self.ax_activations.set_xlabel('Neuron Index (first 50)')
            self.ax_activations.set_ylabel('Layer')
            
            # Create heatmap data
            max_neurons = 50
            heatmap_data = []
            layer_labels = []
            
            for act in self.activation_data:
                activations = act['activations'][:max_neurons]
                heatmap_data.append(activations)
                layer_labels.append(f"L{act['layer_id']}")
            
            if heatmap_data:
                im = self.ax_activations.imshow(heatmap_data, aspect='auto', cmap='viridis')
                self.ax_activations.set_yticks(range(len(layer_labels)))
                self.ax_activations.set_yticklabels(layer_labels)
                plt.colorbar(im, ax=self.ax_activations, label='Activation Value')
        
        # Update network architecture visualization
        self.ax_network.clear()
        self.ax_network.axis('off')
        self.ax_network.set_title('Network Architecture & Activity')
        
        if self.activation_data:
            # Draw simplified network diagram
            layer_sizes = [len(act['activations']) for act in self.activation_data]
            max_size = max(layer_sizes) if layer_sizes else 1
            
            for i, size in enumerate(layer_sizes):
                x = i / (len(layer_sizes) - 1) if len(layer_sizes) > 1 else 0.5
                
                # Draw nodes
                for j in range(min(size, 10)):  # Limit to 10 nodes per layer for visualization
                    y = (j + 1) / (min(size, 10) + 1)
                    
                    # Color based on activation strength
                    if i < len(self.activation_data):
                        act_val = self.activation_data[i]['activations'][j] if j < len(self.activation_data[i]['activations']) else 0
                        color_intensity = min(1.0, abs(act_val))
                        color = plt.cm.RdYlGn(0.5 + color_intensity * 0.5 if act_val > 0 else 0.5 - color_intensity * 0.5)
                    else:
                        color = 'lightgray'
                    
                    circle = plt.Circle((x, y), 0.02, color=color, ec='black', linewidth=1)
                    self.ax_network.add_patch(circle)
                
                # Add layer label
                self.ax_network.text(x, -0.05, f'Layer {i}\n({size} neurons)', 
                                    ha='center', va='top', fontsize=8)
        
        self.ax_network.set_xlim(-0.1, 1.1)
        self.ax_network.set_ylim(-0.15, 1.05)
        
        # Update stats text
        if len(self.epochs) > 0:
            current_epoch = self.epochs[-1]
            current_loss = self.train_losses[-1]
            current_acc = self.test_accuracies[-1]
            
            stats = f"Epoch: {current_epoch}\n"
            stats += f"Loss: {current_loss:.4f}\n"
            stats += f"Accuracy: {current_acc:.2f}%\n"
            
            if len(self.train_losses) > 1:
                loss_change = self.train_losses[-1] - self.train_losses[-2]
                stats += f"Loss Δ: {loss_change:+.4f}\n"
            
            if self.weight_data:
                total_params = sum(w['weights'].size for w in self.weight_data)
                stats += f"Parameters: {total_params:,}"
            
            self.stats_text.set_text(stats)
        
        # Refresh the display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def close(self):
        """Close the visualization window"""
        plt.ioff()
        plt.close(self.fig)
    
    def save_figure(self, filename='training_visualization.png'):
        """Save the current visualization to a file"""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {filename}")

# Made with Bob
