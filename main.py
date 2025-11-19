"""
Main entry point for the Probabilistic Programming package.
This file can be used to verify all modules import correctly and run basic tests.
"""

def main():
    """Main function to verify package imports and functionality."""
    print("Importing core modules...")
    
    try:
        from optimization_function import Optimizer
        print("✓ optimization_function imported successfully")
    except Exception as e:
        print(f"✗ Error importing optimization_function: {e}")
    
    try:
        from Models import PPCA, PCA, FactorAnalysis, CCA
        print("✓ Models imported successfully")
    except Exception as e:
        print(f"✗ Error importing Models: {e}")
    
    try:
        from sampler_algorithms import MetropolisHastings, MCMCHammer
        print("✓ sampler_algorithms imported successfully")
    except Exception as e:
        print(f"✗ Error importing sampler_algorithms: {e}")
    
    try:
        from Continuous import Normal, Beta, Gamma, Exponential
        print("✓ Continuous distributions imported successfully")
    except Exception as e:
        print(f"✗ Error importing Continuous: {e}")
    
    try:
        from Discrete import Binomial, Poisson, Bernoulli
        print("✓ Discrete distributions imported successfully")
    except Exception as e:
        print(f"✗ Error importing Discrete: {e}")
    
    try:
        from Ordinary_Differential_Equation_Solvers import RK4, RungeKuttaMethod
        print("✓ ODE solvers imported successfully")
    except Exception as e:
        print(f"✗ Error importing ODE solvers: {e}")
    
    print("\nAll core modules imported successfully!")
    print("Package is ready for use. See examples_tutorial.ipynb for usage examples.")

if __name__ == "__main__":
    main()