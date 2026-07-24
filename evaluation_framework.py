"""
Neural Network Evaluation Framework

A comprehensive framework for evaluating neural network performance across three dimensions:
1. Performance — How well it learns (accuracy, loss, generalization)
2. Efficiency — Speed and computational requirements
3. Resources — Memory usage and model size

Each metric includes both the "What" (the numbers) and "Why" (the reasoning).
"""

import numpy as np
import time
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================================================
# DATA STRUCTURES FOR EVALUATION RESULTS
# ============================================================================

@dataclass
class PerformanceMetrics:
    """
    Performance Metrics — How well the model learns
    
    What: The numbers that measure learning quality
    Why: Understanding model effectiveness and generalization
    """
    
    # Training metrics
    train_accuracy: float = 0.0
    train_loss: float = 0.0
    
    # Validation/Test metrics
    test_accuracy: float = 0.0
    test_loss: float = 0.0
    
    # Per-class performance
    per_class_accuracy: Dict[int, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    
    # Generalization metrics
    overfitting_gap: float = 0.0  # train_acc - test_acc
    loss_convergence_rate: float = 0.0
    
    # Learning dynamics
    epochs_to_convergence: int = 0
    best_epoch: int = 0
    final_learning_rate: float = 0.0
    
    # Reasoning/Context
    reasoning: Dict[str, str] = field(default_factory=dict)
    
    def add_reasoning(self, metric: str, explanation: str):
        """Add explanation for why a metric matters"""
        self.reasoning[metric] = explanation


@dataclass
class EfficiencyMetrics:
    """
    Efficiency Metrics — Speed and computational requirements
    
    What: Time and compute measurements
    Why: Understanding practical deployment constraints
    """
    
    # Training efficiency
    total_training_time: float = 0.0  # seconds
    time_per_epoch: float = 0.0
    time_per_batch: float = 0.0
    samples_per_second: float = 0.0
    
    # Inference efficiency
    inference_time_single: float = 0.0  # seconds per sample
    inference_time_batch: float = 0.0  # seconds per batch
    throughput: float = 0.0  # samples/second
    
    # Computational complexity
    forward_pass_flops: int = 0  # Floating point operations
    backward_pass_flops: int = 0
    total_flops_per_sample: int = 0
    
    # Convergence efficiency
    iterations_to_target_accuracy: int = 0
    compute_to_accuracy_ratio: float = 0.0
    
    # Reasoning/Context
    reasoning: Dict[str, str] = field(default_factory=dict)
    
    def add_reasoning(self, metric: str, explanation: str):
        """Add explanation for why a metric matters"""
        self.reasoning[metric] = explanation


@dataclass
class ResourceMetrics:
    """
    Resource Metrics — Memory usage and model size
    
    What: Memory and storage measurements
    Why: Understanding deployment feasibility and scalability
    """
    
    # Model size
    total_parameters: int = 0
    trainable_parameters: int = 0
    model_size_mb: float = 0.0
    
    # Memory usage during training
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    memory_per_batch: float = 0.0
    
    # Memory breakdown by component
    weights_memory_mb: float = 0.0
    activations_memory_mb: float = 0.0
    gradients_memory_mb: float = 0.0
    optimizer_state_mb: float = 0.0
    
    # Memory efficiency
    memory_per_parameter: float = 0.0
    activation_memory_ratio: float = 0.0  # activations / weights
    
    # Reasoning/Context
    reasoning: Dict[str, str] = field(default_factory=dict)
    
    def add_reasoning(self, metric: str, explanation: str):
        """Add explanation for why a metric matters"""
        self.reasoning[metric] = explanation


@dataclass
class ArchitectureAnalysis:
    """
    Architecture Analysis — Design choices and their impact
    
    What: Architectural decisions made
    Why: Understanding trade-offs and design rationale
    """
    
    # Architecture description
    layer_config: List[Any] = field(default_factory=list)
    total_layers: int = 0
    hidden_layers: int = 0
    
    # Layer-wise analysis
    layer_sizes: List[int] = field(default_factory=list)
    activation_functions: List[str] = field(default_factory=list)
    regularization_techniques: List[str] = field(default_factory=list)
    
    # Design choices
    has_batch_norm: bool = False
    has_dropout: bool = False
    dropout_rate: float = 0.0
    
    # Capacity analysis
    network_width: int = 0  # Max layer size
    network_depth: int = 0  # Number of hidden layers
    bottleneck_size: int = 0  # Smallest hidden layer
    
    # Trade-offs made
    trade_offs: Dict[str, str] = field(default_factory=dict)
    
    def add_trade_off(self, decision: str, rationale: str):
        """Document architectural trade-offs"""
        self.trade_offs[decision] = rationale


@dataclass
class EvaluationReport:
    """
    Complete Evaluation Report
    
    Combines all metrics with comprehensive reasoning
    """
    
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)
    resources: ResourceMetrics = field(default_factory=ResourceMetrics)
    architecture: ArchitectureAnalysis = field(default_factory=ArchitectureAnalysis)
    
    # Overall assessment
    overall_score: float = 0.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    timestamp: str = ""
    
    def generate_summary(self) -> str:
        """Generate human-readable summary of evaluation"""
        summary = []
        summary.append("=" * 70)
        summary.append("NEURAL NETWORK EVALUATION SUMMARY")
        summary.append("=" * 70)
        
        # Performance section
        summary.append("\n📊 PERFORMANCE")
        summary.append(f"  Test Accuracy:     {self.performance.test_accuracy*100:.2f}%")
        summary.append(f"  Train Accuracy:    {self.performance.train_accuracy*100:.2f}%")
        summary.append(f"  Test Loss:         {self.performance.test_loss:.4f}")
        summary.append(f"  Overfitting Gap:   {self.performance.overfitting_gap*100:.2f}%")
        
        # Efficiency section
        summary.append("\n⚡ EFFICIENCY")
        summary.append(f"  Inference Time:    {self.efficiency.inference_time_single*1000:.2f} ms/sample")
        summary.append(f"  Throughput:        {self.efficiency.throughput:.0f} samples/sec")
        summary.append(f"  Training Time:     {self.efficiency.total_training_time:.2f} seconds")
        
        # Resources section
        summary.append("\n💾 RESOURCES")
        summary.append(f"  Total Parameters:  {self.resources.total_parameters:,}")
        summary.append(f"  Model Size:        {self.resources.model_size_mb:.2f} MB")
        summary.append(f"  Peak Memory:       {self.resources.peak_memory_mb:.2f} MB")
        
        # Architecture section
        summary.append("\n🏗️  ARCHITECTURE")
        summary.append(f"  Total Layers:      {self.architecture.total_layers}")
        summary.append(f"  Network Depth:     {self.architecture.network_depth}")
        summary.append(f"  Network Width:     {self.architecture.network_width}")
        summary.append(f"  Has BatchNorm:     {self.architecture.has_batch_norm}")
        summary.append(f"  Has Dropout:       {self.architecture.has_dropout}")
        
        # Overall assessment
        summary.append(f"\n🎯 OVERALL SCORE: {self.overall_score:.2f}/100")
        
        if self.strengths:
            summary.append("\n✅ STRENGTHS:")
            for strength in self.strengths:
                summary.append(f"  • {strength}")
        
        if self.weaknesses:
            summary.append("\n⚠️  WEAKNESSES:")
            for weakness in self.weaknesses:
                summary.append(f"  • {weakness}")
        
        if self.recommendations:
            summary.append("\n💡 RECOMMENDATIONS:")
            for rec in self.recommendations:
                summary.append(f"  • {rec}")
        
        summary.append("\n" + "=" * 70)
        
        return "\n".join(summary)


# ============================================================================
# EVALUATION FRAMEWORK
# ============================================================================

class NeuralNetworkEvaluator:
    """
    Comprehensive Neural Network Evaluation Framework
    
    Evaluates networks across three dimensions:
    - Performance: Learning quality and accuracy
    - Efficiency: Speed and computational cost
    - Resources: Memory and model size
    """
    
    def __init__(self, network, train_data, test_data, config: Optional[Dict] = None):
        """
        Initialize evaluator
        
        Args:
            network: Neural network instance to evaluate
            train_data: Tuple of (X_train, Y_train)
            test_data: Tuple of (X_test, Y_test)
            config: Optional configuration for evaluation
        """
        self.network = network
        self.X_train, self.Y_train = train_data
        self.X_test, self.Y_test = test_data
        self.config = config or {}
        
        self.report = EvaluationReport()
        self.training_history = defaultdict(list)
    
    # ========================================================================
    # PERFORMANCE EVALUATION
    # ========================================================================
    
    def evaluate_performance(self) -> PerformanceMetrics:
        """
        Evaluate model performance metrics
        
        Returns:
            PerformanceMetrics with all performance measurements
        """
        metrics = PerformanceMetrics()
        
        self.network.eval()
        
        metrics.train_accuracy = self._calculate_accuracy(self.X_train, self.Y_train)
        metrics.test_accuracy = self._calculate_accuracy(self.X_test, self.Y_test)
        
        metrics.train_loss = self._calculate_loss(self.X_train, self.Y_train)
        metrics.test_loss = self._calculate_loss(self.X_test, self.Y_test)
        
        metrics.overfitting_gap = metrics.train_accuracy - metrics.test_accuracy
        metrics.per_class_accuracy = self._calculate_per_class_accuracy()
        metrics.confusion_matrix = self._generate_confusion_matrix()
        
        if self.training_history:
            metrics.epochs_to_convergence = len(self.training_history.get('loss', []))
            if self.training_history.get('accuracy'):
                best_idx = np.argmax(self.training_history['accuracy'])
                metrics.best_epoch = int(best_idx) + 1
        
        metrics.add_reasoning(
            "test_accuracy",
            "Primary metric for model effectiveness on unseen data"
        )
        metrics.add_reasoning(
            "overfitting_gap",
            "Measures generalization: large gap indicates overfitting"
        )
        metrics.add_reasoning(
            "per_class_accuracy",
            "Identifies which classes the model struggles with"
        )
        
        return metrics
    
    def _calculate_accuracy(self, X, Y) -> float:
        """Calculate accuracy on given data"""
        self.network.eval()
        predictions = self.network.forward(X)
        pred_classes = np.argmax(predictions, axis=1)
        actual_classes = np.argmax(Y, axis=1)
        accuracy = np.mean(pred_classes == actual_classes)
        return float(accuracy)
    
    def _calculate_loss(self, X, Y) -> float:
        """Calculate loss on given data"""
        from NeuralNetwork import categorical_cross_entropy
        self.network.eval()
        predictions = self.network.forward(X)
        loss = categorical_cross_entropy(predictions, Y)
        return float(loss)
    
    def _calculate_per_class_accuracy(self) -> Dict[int, float]:
        """Calculate accuracy for each class"""
        self.network.eval()
        predictions = self.network.forward(self.X_test)
        pred_classes = np.argmax(predictions, axis=1)
        actual_classes = np.argmax(self.Y_test, axis=1)
        
        per_class_acc = {}
        n_classes = self.Y_test.shape[1]
        
        for class_id in range(n_classes):
            mask = actual_classes == class_id
            if mask.sum() > 0:
                class_accuracy = np.mean(pred_classes[mask] == actual_classes[mask])
                per_class_acc[class_id] = float(class_accuracy)
            else:
                per_class_acc[class_id] = 0.0
        
        return per_class_acc
    
    def _generate_confusion_matrix(self) -> np.ndarray:
        """Generate confusion matrix"""
        self.network.eval()
        predictions = self.network.forward(self.X_test)
        pred_classes = np.argmax(predictions, axis=1)
        actual_classes = np.argmax(self.Y_test, axis=1)
        
        n_classes = self.Y_test.shape[1]
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        
        for actual, pred in zip(actual_classes, pred_classes):
            confusion[actual, pred] += 1
        
        return confusion
    
    # ========================================================================
    # EFFICIENCY EVALUATION
    # ========================================================================
    
    def evaluate_efficiency(self, num_samples: int = 1000) -> EfficiencyMetrics:
        """
        Evaluate computational efficiency
        
        Args:
            num_samples: Number of samples for timing tests
            
        Returns:
            EfficiencyMetrics with all efficiency measurements
        """
        metrics = EfficiencyMetrics()
        
        # Measure inference time for single sample
        metrics.inference_time_single = self._measure_inference_time(batch_size=1)
        
        # Measure inference time for batch
        metrics.inference_time_batch = self._measure_inference_time(batch_size=64)
        
        # Calculate throughput
        if metrics.inference_time_batch > 0:
            metrics.throughput = 64 / metrics.inference_time_batch
        
        # Estimate FLOPs
        metrics.forward_pass_flops = self._estimate_forward_flops()
        metrics.backward_pass_flops = self._estimate_backward_flops()
        metrics.total_flops_per_sample = metrics.forward_pass_flops + metrics.backward_pass_flops
        
        # Add reasoning
        metrics.add_reasoning(
            "inference_time_single",
            "Critical for real-time applications and user experience"
        )
        metrics.add_reasoning(
            "throughput",
            "Determines scalability and batch processing capability"
        )
        metrics.add_reasoning(
            "total_flops_per_sample",
            "Indicates computational complexity and hardware requirements"
        )
        
        return metrics
    
    def _measure_inference_time(self, batch_size: int, num_runs: int = 100) -> float:
        """Measure average inference time"""
        self.network.eval()
        
        # Prepare test batch
        test_batch = self.X_test[:batch_size]
        
        # Warmup run
        _ = self.network.forward(test_batch)
        
        # Timed runs
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.network.forward(test_batch)
            end = time.time()
            times.append(end - start)
        
        return float(np.mean(times))
    
    def _estimate_forward_flops(self) -> int:
        """Estimate FLOPs for forward pass"""
        total_flops = 0
        
        for layer in self.network.layers:
            if hasattr(layer, 'W'):
                # Matrix multiplication: 2 * m * n * p (for m x n @ n x p)
                # For layer: batch_size * input_size * output_size
                input_size = layer.W.shape[0]
                output_size = layer.W.shape[1]
                # Approximate for single sample
                flops = 2 * input_size * output_size
                total_flops += flops
        
        return int(total_flops)
    
    def _estimate_backward_flops(self) -> int:
        """Estimate FLOPs for backward pass"""
        # Backward pass is approximately 2x forward pass
        # (gradient computation + weight updates)
        return int(self._estimate_forward_flops() * 2)
    
    # ========================================================================
    # RESOURCE EVALUATION
    # ========================================================================
    
    def evaluate_resources(self) -> ResourceMetrics:
        """
        Evaluate resource usage
        
        Returns:
            ResourceMetrics with all resource measurements
        """
        metrics = ResourceMetrics()
        
        # Count parameters
        metrics.total_parameters = self._count_parameters()
        metrics.trainable_parameters = metrics.total_parameters  # All params are trainable
        
        # Calculate model size
        metrics.model_size_mb = self._calculate_model_size()
        
        # Measure memory usage
        metrics.peak_memory_mb = self._measure_peak_memory()
        
        # Break down memory by component
        metrics.weights_memory_mb = metrics.model_size_mb
        metrics.activations_memory_mb = self._estimate_activations_memory()
        metrics.gradients_memory_mb = metrics.weights_memory_mb  # Same size as weights
        metrics.optimizer_state_mb = metrics.weights_memory_mb * 2  # Adam stores m and v
        
        # Calculate derived metrics
        if metrics.total_parameters > 0:
            metrics.memory_per_parameter = metrics.model_size_mb / metrics.total_parameters * 1024 * 1024  # bytes
        
        if metrics.weights_memory_mb > 0:
            metrics.activation_memory_ratio = metrics.activations_memory_mb / metrics.weights_memory_mb
        
        # Add reasoning
        metrics.add_reasoning(
            "total_parameters",
            "Determines model capacity and minimum memory requirements"
        )
        metrics.add_reasoning(
            "peak_memory_mb",
            "Critical for deployment on resource-constrained devices"
        )
        metrics.add_reasoning(
            "activation_memory_ratio",
            "High ratio indicates memory bottleneck during training"
        )
        
        return metrics
    
    def _count_parameters(self) -> int:
        """Count total parameters in network"""
        total = 0
        for layer in self.network.layers:
            if hasattr(layer, 'W'):
                total += layer.W.size + layer.b.size
            elif hasattr(layer, 'gamma'):  # BatchNorm
                total += layer.gamma.size + layer.beta.size
        return int(total)
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in MB"""
        # Assuming float32 (4 bytes per parameter)
        bytes_per_param = 4
        total_params = self._count_parameters()
        size_mb = (total_params * bytes_per_param) / (1024 * 1024)
        return float(size_mb)
    
    def _measure_peak_memory(self) -> float:
        """Measure peak memory usage"""
        # Estimate peak memory during forward pass
        # This is weights + activations + gradients + optimizer state
        weights_mb = self._calculate_model_size()
        activations_mb = self._estimate_activations_memory()
        gradients_mb = weights_mb
        optimizer_mb = weights_mb * 2  # Adam stores m and v
        
        peak = weights_mb + activations_mb + gradients_mb + optimizer_mb
        return float(peak)
    
    def _estimate_activations_memory(self) -> float:
        """Estimate memory for activations"""
        # Estimate based on largest layer and batch size
        max_layer_size = 0
        for layer in self.network.layers:
            if hasattr(layer, 'W'):
                max_layer_size = max(max_layer_size, layer.W.shape[1])
        
        # Assume batch size of 64 and float32
        batch_size = 64
        bytes_per_value = 4
        activation_mb = (batch_size * max_layer_size * bytes_per_value) / (1024 * 1024)
        
        # Multiply by number of layers (stored during backprop)
        num_layers = len([l for l in self.network.layers if hasattr(l, 'W')])
        return float(activation_mb * num_layers)
    
    # ========================================================================
    # ARCHITECTURE ANALYSIS
    # ========================================================================
    
    def analyze_architecture(self) -> ArchitectureAnalysis:
        """
        Analyze network architecture and design choices
        
        Returns:
            ArchitectureAnalysis with architectural insights
        """
        analysis = ArchitectureAnalysis()
        
        # Extract layer configuration
        analysis.total_layers = len(self.network.layers)
        
        # Analyze layer sizes
        layer_sizes = []
        activation_funcs = []
        regularization = []
        
        for layer in self.network.layers:
            if hasattr(layer, 'W'):
                layer_sizes.append(layer.W.shape[1])
                # Determine activation function
                if hasattr(layer, 'softmax'):
                    activation_funcs.append('softmax')
                elif hasattr(layer, 'sigmoid'):
                    activation_funcs.append('sigmoid')
                else:
                    activation_funcs.append('relu')
            elif hasattr(layer, 'rate'):  # Dropout
                regularization.append('dropout')
                analysis.has_dropout = True
                analysis.dropout_rate = layer.rate
            elif hasattr(layer, 'gamma'):  # BatchNorm
                regularization.append('batchnorm')
                analysis.has_batch_norm = True
        
        analysis.layer_sizes = layer_sizes
        analysis.activation_functions = activation_funcs
        analysis.regularization_techniques = regularization
        
        # Calculate network properties
        if layer_sizes:
            analysis.network_width = max(layer_sizes)
            analysis.network_depth = len(layer_sizes) - 1  # Exclude output layer
            analysis.bottleneck_size = min(layer_sizes[:-1]) if len(layer_sizes) > 1 else layer_sizes[0]
        
        analysis.hidden_layers = len(layer_sizes) - 1 if layer_sizes else 0
        
        # Document trade-offs
        analysis.add_trade_off(
            "depth_vs_width",
            f"Network has {analysis.network_depth} hidden layers with max width {analysis.network_width}. "
            "Deeper networks learn hierarchical features but are harder to train"
        )
        analysis.add_trade_off(
            "regularization",
            f"Uses {', '.join(regularization) if regularization else 'no regularization'}. "
            "Dropout/BatchNorm prevent overfitting but add computational cost"
        )
        
        if analysis.bottleneck_size < analysis.network_width:
            analysis.add_trade_off(
                "bottleneck",
                f"Bottleneck layer of size {analysis.bottleneck_size} forces dimensionality reduction, "
                "which can improve generalization but may lose information"
            )
        
        return analysis
    
    # ========================================================================
    # COMPREHENSIVE EVALUATION
    # ========================================================================
    
    def run_full_evaluation(self) -> EvaluationReport:
        """
        Run complete evaluation across all dimensions
        
        Returns:
            Complete EvaluationReport with all metrics and analysis
        """
        print("=" * 70)
        print("Running Comprehensive Neural Network Evaluation")
        print("=" * 70)
        
        # Evaluate performance
        print("\n[1/4] Evaluating Performance Metrics...")
        self.report.performance = self.evaluate_performance()
        
        # Evaluate efficiency
        print("[2/4] Evaluating Efficiency Metrics...")
        self.report.efficiency = self.evaluate_efficiency()
        
        # Evaluate resources
        print("[3/4] Evaluating Resource Usage...")
        self.report.resources = self.evaluate_resources()
        
        # Analyze architecture
        print("[4/4] Analyzing Architecture...")
        self.report.architecture = self.analyze_architecture()
        
        # Generate overall assessment
        print("\nGenerating Overall Assessment...")
        self._generate_overall_assessment()
        
        print("\n" + "=" * 70)
        print("Evaluation Complete!")
        print("=" * 70)
        
        return self.report
    
    def _generate_overall_assessment(self):
        """Generate overall assessment with strengths, weaknesses, recommendations"""
        perf = self.report.performance
        eff = self.report.efficiency
        res = self.report.resources
        arch = self.report.architecture
        
        # Calculate overall score (0-100)
        score = 0.0
        
        # Performance component (40%)
        if perf.test_accuracy > 0:
            score += perf.test_accuracy * 40
        
        # Efficiency component (30%)
        if eff.throughput > 0:
            # Normalize throughput (assume 1000 samples/sec is good)
            eff_score = min(eff.throughput / 1000, 1.0) * 30
            score += eff_score
        
        # Resource efficiency component (30%)
        if res.total_parameters > 0:
            # Penalize very large models (assume 1M params is baseline)
            param_score = max(0, 30 - (res.total_parameters / 1_000_000) * 5)
            score += param_score
        
        self.report.overall_score = score
        
        # Identify strengths
        if perf.test_accuracy > 0.90:
            self.report.strengths.append(f"High accuracy ({perf.test_accuracy*100:.1f}%)")
        
        if perf.overfitting_gap < 0.05:
            self.report.strengths.append("Good generalization (low overfitting)")
        
        if eff.throughput > 500:
            self.report.strengths.append(f"Fast inference ({eff.throughput:.0f} samples/sec)")
        
        if res.model_size_mb < 10:
            self.report.strengths.append(f"Compact model ({res.model_size_mb:.2f} MB)")
        
        # Identify weaknesses
        if perf.test_accuracy < 0.80:
            self.report.weaknesses.append(f"Low accuracy ({perf.test_accuracy*100:.1f}%)")
        
        if perf.overfitting_gap > 0.10:
            self.report.weaknesses.append(f"Overfitting detected (gap: {perf.overfitting_gap*100:.1f}%)")
        
        if eff.inference_time_single > 0.1:
            self.report.weaknesses.append(f"Slow inference ({eff.inference_time_single*1000:.1f} ms/sample)")
        
        if res.model_size_mb > 50:
            self.report.weaknesses.append(f"Large model size ({res.model_size_mb:.1f} MB)")
        
        # Generate recommendations
        if perf.overfitting_gap > 0.10:
            self.report.recommendations.append("Add more regularization (dropout, L2) to reduce overfitting")
        
        if perf.test_accuracy < 0.85:
            self.report.recommendations.append("Consider increasing model capacity or training longer")
        
        if eff.inference_time_single > 0.05:
            self.report.recommendations.append("Optimize model for inference (pruning, quantization)")
        
        if not arch.has_batch_norm and arch.network_depth > 2:
            self.report.recommendations.append("Add batch normalization for deeper networks")
        
        if res.model_size_mb > 20:
            self.report.recommendations.append("Consider model compression techniques")
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def print_report(self):
        """Print formatted evaluation report"""
        print(self.report.generate_summary())
        
        # Print detailed reasoning
        print("\n" + "=" * 70)
        print("DETAILED REASONING")
        print("=" * 70)
        
        if self.report.performance.reasoning:
            print("\n📊 Performance Metrics:")
            for metric, reason in self.report.performance.reasoning.items():
                print(f"  • {metric}: {reason}")
        
        if self.report.efficiency.reasoning:
            print("\n⚡ Efficiency Metrics:")
            for metric, reason in self.report.efficiency.reasoning.items():
                print(f"  • {metric}: {reason}")
        
        if self.report.resources.reasoning:
            print("\n💾 Resource Metrics:")
            for metric, reason in self.report.resources.reasoning.items():
                print(f"  • {metric}: {reason}")
        
        if self.report.architecture.trade_offs:
            print("\n🏗️  Architecture Trade-offs:")
            for decision, rationale in self.report.architecture.trade_offs.items():
                print(f"  • {decision}: {rationale}")
    
    def save_report(self, filename: str = "evaluation_report.json"):
        """Save evaluation report to file"""
        import json
        from datetime import datetime
        
        self.report.timestamp = datetime.now().isoformat()
        
        # Convert report to dictionary
        report_dict = {
            'timestamp': self.report.timestamp,
            'overall_score': self.report.overall_score,
            'performance': {
                'train_accuracy': self.report.performance.train_accuracy,
                'test_accuracy': self.report.performance.test_accuracy,
                'train_loss': self.report.performance.train_loss,
                'test_loss': self.report.performance.test_loss,
                'overfitting_gap': self.report.performance.overfitting_gap,
                'per_class_accuracy': self.report.performance.per_class_accuracy,
                'reasoning': self.report.performance.reasoning
            },
            'efficiency': {
                'inference_time_single': self.report.efficiency.inference_time_single,
                'inference_time_batch': self.report.efficiency.inference_time_batch,
                'throughput': self.report.efficiency.throughput,
                'forward_pass_flops': self.report.efficiency.forward_pass_flops,
                'backward_pass_flops': self.report.efficiency.backward_pass_flops,
                'reasoning': self.report.efficiency.reasoning
            },
            'resources': {
                'total_parameters': self.report.resources.total_parameters,
                'model_size_mb': self.report.resources.model_size_mb,
                'peak_memory_mb': self.report.resources.peak_memory_mb,
                'weights_memory_mb': self.report.resources.weights_memory_mb,
                'activations_memory_mb': self.report.resources.activations_memory_mb,
                'reasoning': self.report.resources.reasoning
            },
            'architecture': {
                'total_layers': self.report.architecture.total_layers,
                'hidden_layers': self.report.architecture.hidden_layers,
                'network_width': self.report.architecture.network_width,
                'network_depth': self.report.architecture.network_depth,
                'has_batch_norm': self.report.architecture.has_batch_norm,
                'has_dropout': self.report.architecture.has_dropout,
                'trade_offs': self.report.architecture.trade_offs
            },
            'strengths': self.report.strengths,
            'weaknesses': self.report.weaknesses,
            'recommendations': self.report.recommendations
        }
        
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"\n✅ Report saved to {filename}")
    
    def generate_comparison_report(self, other_report: EvaluationReport) -> str:
        """Generate comparison between two evaluation reports"""
        lines = []
        lines.append("=" * 70)
        lines.append("MODEL COMPARISON REPORT")
        lines.append("=" * 70)
        
        # Performance comparison
        lines.append("\n📊 PERFORMANCE COMPARISON")
        acc_diff = self.report.performance.test_accuracy - other_report.performance.test_accuracy
        lines.append(f"  Test Accuracy: {self.report.performance.test_accuracy*100:.2f}% vs {other_report.performance.test_accuracy*100:.2f}% ({acc_diff*100:+.2f}%)")
        
        # Efficiency comparison
        lines.append("\n⚡ EFFICIENCY COMPARISON")
        if self.report.efficiency.throughput > 0 and other_report.efficiency.throughput > 0:
            throughput_ratio = self.report.efficiency.throughput / other_report.efficiency.throughput
            lines.append(f"  Throughput: {self.report.efficiency.throughput:.0f} vs {other_report.efficiency.throughput:.0f} samples/sec ({throughput_ratio:.2f}x)")
        
        # Resource comparison
        lines.append("\n💾 RESOURCE COMPARISON")
        param_diff = self.report.resources.total_parameters - other_report.resources.total_parameters
        lines.append(f"  Parameters: {self.report.resources.total_parameters:,} vs {other_report.resources.total_parameters:,} ({param_diff:+,})")
        
        return "\n".join(lines)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compare_models(evaluator1: NeuralNetworkEvaluator,
                   evaluator2: NeuralNetworkEvaluator) -> Dict[str, Any]:
    """
    Compare two models across all metrics
    
    Args:
        evaluator1: First model evaluator
        evaluator2: Second model evaluator
        
    Returns:
        Comparison results with winner for each metric
    """
    # Run evaluations if not already done
    if not evaluator1.report.performance.test_accuracy:
        evaluator1.run_full_evaluation()
    if not evaluator2.report.performance.test_accuracy:
        evaluator2.run_full_evaluation()
    
    comparison = {
        'performance': {
            'accuracy_winner': 'model1' if evaluator1.report.performance.test_accuracy > evaluator2.report.performance.test_accuracy else 'model2',
            'model1_accuracy': evaluator1.report.performance.test_accuracy,
            'model2_accuracy': evaluator2.report.performance.test_accuracy,
            'accuracy_difference': abs(evaluator1.report.performance.test_accuracy - evaluator2.report.performance.test_accuracy)
        },
        'efficiency': {
            'speed_winner': 'model1' if evaluator1.report.efficiency.throughput > evaluator2.report.efficiency.throughput else 'model2',
            'model1_throughput': evaluator1.report.efficiency.throughput,
            'model2_throughput': evaluator2.report.efficiency.throughput
        },
        'resources': {
            'size_winner': 'model1' if evaluator1.report.resources.total_parameters < evaluator2.report.resources.total_parameters else 'model2',
            'model1_parameters': evaluator1.report.resources.total_parameters,
            'model2_parameters': evaluator2.report.resources.total_parameters
        },
        'overall_winner': 'model1' if evaluator1.report.overall_score > evaluator2.report.overall_score else 'model2'
    }
    
    return comparison


def benchmark_against_baseline(evaluator: NeuralNetworkEvaluator,
                               baseline_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Compare model against baseline metrics
    
    Args:
        evaluator: Model evaluator
        baseline_metrics: Dictionary of baseline metric values
        
    Returns:
        Percentage improvements over baseline
    """
    if not evaluator.report.performance.test_accuracy:
        evaluator.run_full_evaluation()
    
    improvements = {}
    
    # Compare accuracy
    if 'accuracy' in baseline_metrics:
        baseline_acc = baseline_metrics['accuracy']
        current_acc = evaluator.report.performance.test_accuracy
        improvements['accuracy'] = ((current_acc - baseline_acc) / baseline_acc) * 100
    
    # Compare throughput
    if 'throughput' in baseline_metrics:
        baseline_throughput = baseline_metrics['throughput']
        current_throughput = evaluator.report.efficiency.throughput
        if baseline_throughput > 0:
            improvements['throughput'] = ((current_throughput - baseline_throughput) / baseline_throughput) * 100
    
    # Compare model size
    if 'parameters' in baseline_metrics:
        baseline_params = baseline_metrics['parameters']
        current_params = evaluator.report.resources.total_parameters
        improvements['parameters'] = ((baseline_params - current_params) / baseline_params) * 100  # Negative means larger
    
    return improvements


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    """
    Example usage of the evaluation framework
    
    This demonstrates how to use the framework to evaluate a trained network
    """
    
    print("=" * 70)
    print("NEURAL NETWORK EVALUATION FRAMEWORK")
    print("=" * 70)
    print("\nThis framework provides comprehensive evaluation across:")
    print("  ✓ Performance Metrics (accuracy, loss, generalization)")
    print("  ✓ Efficiency Metrics (speed, compute, throughput)")
    print("  ✓ Resource Metrics (memory, model size)")
    print("  ✓ Architecture Analysis (design choices, trade-offs)")
    print("\nEach metric includes both 'What' (numbers) and 'Why' (reasoning)")
    
    print("\n" + "=" * 70)
    print("EXAMPLE USAGE")
    print("=" * 70)
    print("""
from evaluation_framework import NeuralNetworkEvaluator
from NeuralNetwork import Network, X_train, Y_train, X_test, Y_test

# Create and train your network
net = Network([784, 128, 'batchnorm', 'dropout', 64, 10], output='softmax')
# ... train the network ...

# Create evaluator
evaluator = NeuralNetworkEvaluator(
    network=net,
    train_data=(X_train, Y_train),
    test_data=(X_test, Y_test)
)

# Run full evaluation
report = evaluator.run_full_evaluation()

# Print detailed report
evaluator.print_report()

# Save report to file
evaluator.save_report('my_model_evaluation.json')

# Compare with baseline
baseline = {'accuracy': 0.85, 'throughput': 500, 'parameters': 100000}
improvements = benchmark_against_baseline(evaluator, baseline)
print(f"Improvements over baseline: {improvements}")
    """)
    
    print("\n" + "=" * 70)
    print("Ready to use! Import and evaluate your neural networks.")
    print("=" * 70)


