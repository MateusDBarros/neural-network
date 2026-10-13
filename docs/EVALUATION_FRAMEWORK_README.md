# Neural Network Evaluation Framework

A comprehensive framework for evaluating neural networks across three critical dimensions: **Performance**, **Efficiency**, and **Resources**. Each metric includes both the measurement (What) and the reasoning behind it (Why).

## 🎯 Overview

This framework provides:
- **Performance Metrics** — How well the model learns
- **Efficiency Metrics** — Speed and computational requirements  
- **Resource Metrics** — Memory usage and model size
- **Architecture Analysis** — Design choices and trade-offs

## 📦 Installation

No additional dependencies required beyond the base neural network implementation:
```python
import numpy as np
from evaluation_framework import NeuralNetworkEvaluator
```

## 🚀 Quick Start

```python
from evaluation_framework import NeuralNetworkEvaluator
from NeuralNetwork import Network, X_train, Y_train, X_test, Y_test

# Train your network
net = Network([784, 128, 64, 10], output='softmax')
# ... training code ...

# Create evaluator
evaluator = NeuralNetworkEvaluator(
    network=net,
    train_data=(X_train, Y_train),
    test_data=(X_test, Y_test)
)

# Run full evaluation
report = evaluator.run_full_evaluation()

# Print results
evaluator.print_report()

# Save to JSON
evaluator.save_report('evaluation.json')
```

## 📊 Metrics Explained

### Performance Metrics (What + Why)

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **Test Accuracy** | Correct predictions on unseen data | Primary indicator of model effectiveness |
| **Train Accuracy** | Correct predictions on training data | Helps identify overfitting |
| **Overfitting Gap** | Difference between train and test accuracy | Large gap indicates poor generalization |
| **Per-Class Accuracy** | Accuracy for each individual class | Identifies which classes are problematic |
| **Confusion Matrix** | Detailed prediction breakdown | Shows specific misclassification patterns |
| **Loss Values** | Training and test loss | Indicates convergence quality |

### Efficiency Metrics (What + Why)

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **Inference Time (Single)** | Time per sample (ms) | Critical for real-time applications |
| **Inference Time (Batch)** | Time per batch | Important for batch processing |
| **Throughput** | Samples processed per second | Determines scalability |
| **Forward Pass FLOPs** | Floating point operations | Indicates computational complexity |
| **Backward Pass FLOPs** | Training computation cost | Affects training time and hardware needs |

### Resource Metrics (What + Why)

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **Total Parameters** | Number of trainable weights | Determines model capacity and memory |
| **Model Size (MB)** | Storage space required | Critical for deployment constraints |
| **Peak Memory (MB)** | Maximum memory during training | Limits batch size and hardware requirements |
| **Weights Memory** | Memory for model parameters | Base memory requirement |
| **Activations Memory** | Memory for intermediate values | Scales with batch size |
| **Gradients Memory** | Memory for backpropagation | Same as weights during training |
| **Optimizer State** | Memory for Adam's m and v | 2x weights for Adam optimizer |

### Architecture Analysis (What + Why)

| Aspect | What It Analyzes | Why It Matters |
|--------|------------------|----------------|
| **Network Depth** | Number of hidden layers | Deeper = more hierarchical features |
| **Network Width** | Maximum layer size | Wider = more capacity per layer |
| **Bottleneck Size** | Smallest hidden layer | Forces dimensionality reduction |
| **Regularization** | Dropout, BatchNorm detection | Affects overfitting and training stability |
| **Trade-offs** | Design decision documentation | Explains architectural choices |

## 🔍 Detailed Usage

### Basic Evaluation

```python
# Create evaluator
evaluator = NeuralNetworkEvaluator(
    network=trained_network,
    train_data=(X_train, Y_train),
    test_data=(X_test, Y_test)
)

# Run individual evaluations
perf_metrics = evaluator.evaluate_performance()
eff_metrics = evaluator.evaluate_efficiency()
res_metrics = evaluator.evaluate_resources()
arch_analysis = evaluator.analyze_architecture()

# Or run everything at once
report = evaluator.run_full_evaluation()
```

### Accessing Results

```python
# Performance
print(f"Test Accuracy: {report.performance.test_accuracy * 100:.2f}%")
print(f"Overfitting Gap: {report.performance.overfitting_gap * 100:.2f}%")

# Efficiency
print(f"Throughput: {report.efficiency.throughput:.0f} samples/sec")
print(f"Inference Time: {report.efficiency.inference_time_single * 1000:.2f} ms")

# Resources
print(f"Parameters: {report.resources.total_parameters:,}")
print(f"Model Size: {report.resources.model_size_mb:.2f} MB")

# Architecture
print(f"Depth: {report.architecture.network_depth}")
print(f"Width: {report.architecture.network_width}")
```

### Comparing Models

```python
from evaluation_framework import compare_models

evaluator1 = NeuralNetworkEvaluator(network1, train_data, test_data)
evaluator2 = NeuralNetworkEvaluator(network2, train_data, test_data)

comparison = compare_models(evaluator1, evaluator2)
print(f"Winner: {comparison['overall_winner']}")
print(f"Accuracy difference: {comparison['performance']['accuracy_difference']:.4f}")
```

### Baseline Comparison

```python
from evaluation_framework import benchmark_against_baseline

baseline = {
    'accuracy': 0.85,
    'throughput': 500,
    'parameters': 100000
}

improvements = benchmark_against_baseline(evaluator, baseline)
print(f"Accuracy improvement: {improvements['accuracy']:+.2f}%")
print(f"Speed improvement: {improvements['throughput']:+.2f}%")
```

## 📈 Understanding the Overall Score

The framework calculates an overall score (0-100) based on:
- **40%** Performance (test accuracy)
- **30%** Efficiency (throughput normalized to 1000 samples/sec)
- **30%** Resource efficiency (penalizes very large models)

## 💡 Automatic Insights

The framework automatically identifies:

### Strengths
- High accuracy (>90%)
- Good generalization (overfitting gap <5%)
- Fast inference (>500 samples/sec)
- Compact model (<10 MB)

### Weaknesses
- Low accuracy (<80%)
- Overfitting (gap >10%)
- Slow inference (>100ms per sample)
- Large model (>50 MB)

### Recommendations
- Add regularization if overfitting
- Increase capacity if accuracy is low
- Optimize for inference if too slow
- Add batch normalization for deep networks
- Consider compression for large models

## 📄 Report Formats

### Console Output
```python
evaluator.print_report()
```
Produces formatted text with:
- Summary statistics
- Detailed reasoning for each metric
- Architecture trade-offs
- Strengths, weaknesses, recommendations

### JSON Export
```python
evaluator.save_report('evaluation.json')
```
Saves complete evaluation data including:
- All metrics with values
- Reasoning for each metric
- Architecture analysis
- Assessment and recommendations
- Timestamp

## 🎓 Example Output

```
======================================================================
NEURAL NETWORK EVALUATION SUMMARY
======================================================================

📊 PERFORMANCE
  Test Accuracy:     92.50%
  Train Accuracy:    94.20%
  Test Loss:         0.2341
  Overfitting Gap:   1.70%

⚡ EFFICIENCY
  Inference Time:    2.34 ms/sample
  Throughput:        427 samples/sec
  Training Time:     45.67 seconds

💾 RESOURCES
  Total Parameters:  101,770
  Model Size:        0.39 MB
  Peak Memory:       12.45 MB

🏗️  ARCHITECTURE
  Total Layers:      5
  Network Depth:     2
  Network Width:     128
  Has BatchNorm:     True
  Has Dropout:       True

🎯 OVERALL SCORE: 78.5/100

✅ STRENGTHS:
  • High accuracy (92.5%)
  • Good generalization (low overfitting)
  • Compact model (0.39 MB)

💡 RECOMMENDATIONS:
  • Consider increasing model capacity for higher accuracy
  • Optimize inference for real-time applications
```

## 🔧 Advanced Features

### Custom Configuration
```python
evaluator = NeuralNetworkEvaluator(
    network=net,
    train_data=(X_train, Y_train),
    test_data=(X_test, Y_test),
    config={
        'num_timing_runs': 200,  # More accurate timing
        'batch_size': 128        # Different batch size
    }
)
```

### Training History Integration
```python
# Track metrics during training
evaluator.training_history['loss'].append(loss)
evaluator.training_history['accuracy'].append(acc)

# Framework will use this for convergence analysis
report = evaluator.run_full_evaluation()
print(f"Converged at epoch: {report.performance.best_epoch}")
```

## 📚 Complete Example

See `example_evaluation.py` for a complete working example that:
1. Trains a simple network
2. Runs full evaluation
3. Prints detailed report
4. Saves to JSON
5. Compares with baseline
6. Shows all metrics and insights

Run it with:
```bash
python example_evaluation.py
```

## 🤝 Integration with Existing Code

The framework is designed to work seamlessly with the existing neural network implementation:
- Uses the same `Network` class
- Compatible with all layer types (Dense, Dropout, BatchNorm)
- Works with any optimizer
- No modifications to training code required

## 📝 Notes

- All timing measurements include warmup runs for accuracy
- Memory estimates assume float32 (4 bytes per parameter)
- FLOPs estimation is approximate (exact count depends on implementation)
- Batch size for memory estimation defaults to 64
- Confusion matrix and per-class metrics use test set only

## 🎯 Best Practices

1. **Always evaluate after training** — Don't evaluate during training
2. **Use consistent data** — Same train/test split for comparisons
3. **Run multiple times** — Timing can vary, average multiple runs
4. **Save reports** — Keep JSON files for historical comparison
5. **Check reasoning** — Understand why metrics matter for your use case

## 🚀 Future Enhancements

Potential additions:
- GPU memory tracking
- Quantization analysis
- Pruning metrics
- Energy consumption estimation
- Cross-validation support
- Visualization integration

---

**Made with ❤️ for comprehensive neural network evaluation**