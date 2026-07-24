"""
Example: Using the Neural Network Evaluation Framework

This script demonstrates how to evaluate a trained neural network
using the comprehensive evaluation framework.
"""

from evaluation_framework import NeuralNetworkEvaluator, benchmark_against_baseline
from NeuralNetwork import (
    Network, Adam,
    categorical_cross_entropy, categorical_cross_entropy_grad,
    X_train, Y_train, X_test, Y_test,
    get_batches, accuracy
)


def train_simple_network(epochs=5, batch_size=64):
    """Train a simple network for demonstration"""
    print("Training a simple neural network...")
    print("=" * 70)
    
    # Create network
    net = Network([784, 128, 64, 10], output='softmax')
    optimizer = Adam(learning_rate=0.001)
    
    # Training loop
    for epoch in range(epochs):
        net.train()
        total_loss = 0
        batches = 0
        
        for X_batch, Y_batch in get_batches(X_train, Y_train, batch_size):
            predicted = net.forward(X_batch)
            total_loss += categorical_cross_entropy(predicted, Y_batch)
            grad = categorical_cross_entropy_grad(predicted, Y_batch)
            net.backward(grad, optimizer)
            batches += 1
        
        # Evaluate
        net.eval()
        test_pred = net.forward(X_test)
        test_acc = accuracy(test_pred, Y_test)
        avg_loss = total_loss / batches
        
        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}  Accuracy: {test_acc*100:.1f}%")
    
    print("\nTraining complete!")
    return net


def main():
    """Main evaluation example"""
    
    print("\n" + "=" * 70)
    print("NEURAL NETWORK EVALUATION FRAMEWORK - EXAMPLE")
    print("=" * 70)
    
    # Step 1: Train a network
    print("\n[Step 1] Training Network...")
    network = train_simple_network(epochs=5)
    
    # Step 2: Create evaluator
    print("\n[Step 2] Creating Evaluator...")
    evaluator = NeuralNetworkEvaluator(
        network=network,
        train_data=(X_train, Y_train),
        test_data=(X_test, Y_test)
    )
    
    # Step 3: Run full evaluation
    print("\n[Step 3] Running Full Evaluation...")
    report = evaluator.run_full_evaluation()
    
    # Step 4: Print detailed report
    print("\n[Step 4] Evaluation Results:")
    evaluator.print_report()
    
    # Step 5: Save report
    print("\n[Step 5] Saving Report...")
    evaluator.save_report('example_evaluation_report.json')
    
    # Step 6: Compare with baseline
    print("\n[Step 6] Comparing with Baseline...")
    baseline_metrics = {
        'accuracy': 0.85,
        'throughput': 500,
        'parameters': 150000
    }
    
    improvements = benchmark_against_baseline(evaluator, baseline_metrics)
    
    print("\nBaseline Comparison:")
    print("=" * 70)
    for metric, improvement in improvements.items():
        symbol = "📈" if improvement > 0 else "📉"
        print(f"  {symbol} {metric}: {improvement:+.2f}%")
    
    # Step 7: Summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\n✅ Overall Score: {report.overall_score:.1f}/100")
    print(f"✅ Test Accuracy: {report.performance.test_accuracy*100:.2f}%")
    print(f"✅ Throughput: {report.efficiency.throughput:.0f} samples/sec")
    print(f"✅ Model Size: {report.resources.model_size_mb:.2f} MB")
    print(f"✅ Total Parameters: {report.resources.total_parameters:,}")
    
    if report.strengths:
        print("\n💪 Key Strengths:")
        for strength in report.strengths:
            print(f"   • {strength}")
    
    if report.recommendations:
        print("\n💡 Recommendations:")
        for rec in report.recommendations:
            print(f"   • {rec}")
    
    print("\n" + "=" * 70)
    print("Check 'example_evaluation_report.json' for detailed results!")
    print("=" * 70)


if __name__ == '__main__':
    main()

# Made with Bob
