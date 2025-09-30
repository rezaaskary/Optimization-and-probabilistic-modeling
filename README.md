# Optimization and Probabilistic Modeling

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.3.23-orange)](https://github.com/google/jax)
[![TensorFlow Probability](https://img.shields.io/badge/TFP-0.18.0-red)](https://www.tensorflow.org/probability)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)

## Overview

A comprehensive CPU/GPU accelerated Python package for probabilistic programming, Bayesian inference, and optimization. This library provides efficient implementations of various optimization algorithms, probabilistic models, and statistical methods, leveraging JAX for high-performance vectorized computations particularly suited for systems biology and scientific computing applications.

🚧 **Note**: This project is currently under active development.

## Features

### 🔧 Optimization Algorithms
- **Gradient-based optimizers**: SGD, Adam, RMSprop
- **Evolutionary algorithms** (in development)
- Support for both minimization and maximization problems
- Configurable learning rates and convergence criteria

### 📊 Probabilistic Programming
- **Continuous distributions**: Normal, Uniform, Beta, Gamma, Log-normal, Triangular, and more
- **Discrete distributions**: Binomial, Poisson, Categorical, and others
- **MCMC sampling algorithms**: Metropolis-Hastings and variants
- Model parallelization for efficient multi-chain sampling

### 🧬 Statistical Models
- **Principal Component Analysis (PCA)**: Complete and incomplete matrix formulations
- **Probabilistic PCA (PPCA)**: JAX-accelerated implementation with missing data support
- **Feature selection**: Chi-square tests for categorical data
- **Sensitivity analysis**: Distance-based methods with multiple distribution support

### 🔬 Scientific Computing
- **ODE solvers**: Numerical integration for dynamical systems
- **Likelihood estimation**: Various likelihood functions for model fitting
- **Model parallelization**: Efficient evaluation across multiple parameter sets
- **GPU acceleration**: Leveraging JAX's XLA compilation

## Installation

### Prerequisites
- Python 3.9 or higher
- CUDA-compatible GPU (optional, for GPU acceleration)

### Using pip
```bash
git clone https://github.com/rezaaskary/Optimization-and-probabilistic-modeling.git
cd Optimization-and-probabilistic-modeling
pip install -r requirements.txt
```

### Using Docker
```bash
docker build -t optimization-probabilistic .
docker run optimization-probabilistic
```

## Dependencies

- **JAX 0.3.23**: High-performance numerical computing with automatic differentiation
- **TensorFlow Probability 0.18.0**: Probabilistic modeling and inference
- **NumPy 1.23.4**: Fundamental package for scientific computing
- **SciPy**: Additional scientific computing utilities
- **Matplotlib**: Data visualization
- **pandas**: Data manipulation and analysis
- **tqdm**: Progress bars for long-running computations

## Quick Start

### Basic Optimization
```python
from optimization_function import Optimizer

# Create an Adam optimizer
optimizer = Optimizer(algorithm='ADAM', alpha=0.001, 
                     epsilon=1e-8, beta1=0.9, beta2=0.999)

# Use in your optimization loop
for t in range(num_iterations):
    gradients = compute_gradients(parameters)
    parameters = optimizer.fit(parameters, gradients, t)
```

### Probabilistic Modeling
```python
from Probablity_distributions import ContinuousDistributions

# Create a normal distribution
dist = ContinuousDistributions(mu=0, sigma=1, chains=1000)
samples = dist.normal_distribution()
```

### MCMC Sampling
```python
from sampler_algorithms import ModelParallelizer

# Set up parallel model evaluation
model_parallel = ModelParallelizer(
    model=your_model_function,
    chains=4,
    n_obs=100,
    activate_jit=True
)
```

### Principal Component Analysis
```python
from Models import PPCA
import jax.numpy as jnp

# Your data matrix (features × samples)
data = jnp.array(your_data)

# Create PPCA model
ppca = PPCA(y=data, n_comp=3, max_iter=500, tolerance=1e-6)

# Fit the model
ppca.run()
```

## Module Overview

### Core Modules

| Module | Description |
|--------|-------------|
| `optimization_function.py` | Implementation of various optimization algorithms (SGD, Adam, RMSprop) |
| `Models.py` | Statistical models including Probabilistic PCA with missing data support |
| `sampler_algorithms.py` | MCMC sampling algorithms and model parallelization utilities |
| `Probablity_distributions.py` | Comprehensive collection of probability distributions |
| `Discrete.py` | Discrete probability distributions with JAX acceleration |
| `Continuous.py` | Continuous probability distributions |
| `Feature_Selection.py` | Feature selection methods including chi-square tests |
| `Sensitivity_Analysis.py` | Sensitivity analysis tools for model evaluation |
| `Ordinary_Differential_Equation_Solvers.py` | Numerical ODE solvers for dynamical systems |
| `Liklihoods.py` | Likelihood functions for statistical inference |

### Utility Modules

- `main.py`: Entry point and example usage
- `test_file.py`, `testing_II.py`: Test suites and validation scripts
- `requirements.txt`: Package dependencies
- `Dockerfile`: Containerization configuration

## Advanced Features

### GPU Acceleration
The library leverages JAX's XLA compilation for automatic GPU acceleration:
```python
# Enable JIT compilation for faster execution
model_parallel = ModelParallelizer(activate_jit=True)
```

### Missing Data Handling
PPCA implementation supports matrices with missing values:
```python
# Data with NaN values is automatically handled
data_with_missing = jnp.array([[1, 2, jnp.nan], [4, jnp.nan, 6]])
ppca = PPCA(y=data_with_missing, n_comp=2)
```

### Multi-chain Sampling
Efficient parallel sampling across multiple chains:
```python
# Run multiple MCMC chains in parallel
sampler = ModelParallelizer(chains=8, activate_jit=True)
```

## Applications

This library is particularly well-suited for:

- **Systems Biology**: Parameter estimation for biological models
- **Bayesian Inference**: Uncertainty quantification in scientific models
- **Machine Learning**: Dimensionality reduction and feature selection
- **Engineering**: Optimization of complex systems
- **Finance**: Risk modeling and portfolio optimization
- **Scientific Computing**: High-performance numerical simulations

## Development Status

| Component | Status |
|-----------|--------|
| ✅ Optimization algorithms | Complete |
| ✅ Probabilistic distributions | Complete |
| ✅ MCMC sampling | Complete |
| ✅ PCA/PPCA models | Complete |
| ✅ ODE solvers | Complete |
| 🚧 Evolutionary algorithms | In development |
| 🚧 Advanced feature selection | In development |
| 🚧 Documentation | In progress |

## Performance

The library is designed for high-performance computing:
- **JAX acceleration**: Automatic vectorization and GPU support
- **JIT compilation**: Near C-speed performance for numerical computations
- **Memory efficiency**: Optimized for large-scale problems
- **Parallel execution**: Multi-chain and multi-core support

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Testing

Run the test suite:
```bash
python -m unittest discover -s . -p "*test*.py"
```

Or using Docker:
```bash
docker run optimization-probabilistic python -m unittest -s Tests
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{optimization_probabilistic_modeling,
  title={Optimization and Probabilistic Modeling},
  author={Mohammad Reza Askari},
  year={2025},
  url={https://github.com/rezaaskary/Optimization-and-probabilistic-modeling},
  note={Version under development}
}
```

## References

[1] Chib, Siddhartha, and Edward Greenberg. "Understanding the metropolis-hastings algorithm." The american statistician 49.4 (1995): 327-335.

[2] Diwekar, Urmila M. Introduction to applied optimization. Vol. 22. Springer Nature, 2020.

[3] Boyd, Stephen, et al. "Distributed optimization and statistical learning via the alternating direction method of multipliers." Foundations and Trends® in Machine learning 3.1 (2011): 1-122.

[4] Foreman-Mackey, Daniel, et al. "emcee: the MCMC hammer." Publications of the Astronomical Society of the Pacific 125.925 (2013): 306.

[5] Bradbury, James, et al. "JAX: composable transformations of Python+ NumPy programs." Version 0.2 5 (2018): 14-24.

## Contact

**Mohammad Reza Askari**  
📧 maskari@hawk.iit.edu

## Acknowledgments

This project builds upon the excellent work of the JAX and TensorFlow Probability teams, providing GPU-accelerated numerical computing capabilities for the scientific Python ecosystem.
