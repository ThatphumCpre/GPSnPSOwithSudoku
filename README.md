# Sudoku-AI-Bot

## How to run

1. Simply run `sudoku.py`
2. Enter Filename in the program for example `easy1.txt`
3. Watch solving process logs in command window.

# Particle Swarm Optimization Parameters

This document describes the parameters used in a Particle Swarm Optimization (PSO) algorithm.

## General Parameters

- **Swarm Size (`swarm_size`)**:  
  The number of particles in the swarm.  
  **Value**: `100`

- **Mutation Probability (`mutation_prob`)**:  
  The probability of mutation for each particle.  
  **Value**: `0.5`

- **Maximum Iterations (`max_iterations`)**:  
  The maximum number of iterations the algorithm will run.  
  **Value**: `1000`

## Dynamic Coefficient

- **Dynamic Parameter (`dynamic_parameter`)**:  
  Indicates whether the dynamic adjustment of coefficients is enabled.  
  **Value**: `True`

## Cognitive Coefficient (c1)

- **Initial Cognitive Coefficient (`initial_cognitive`)**:  
  The initial value for the cognitive coefficient, representing the particle's tendency to move toward its personal best position.  
  **Value**: `0.1`

- **Final Cognitive Coefficient (`final_cognitive`)**:  
  The final value for the cognitive coefficient after dynamic adjustment.  
  **Value**: `0.5`

- **Cognitive Adjustment Parameter (`cognitive_param`)**:  
  The rate or factor influencing the change in the cognitive coefficient.  
  **Value**: `0.15`

## Social Coefficient (c2)

- **Initial Social Coefficient (`initial_social`)**:  
  The initial value for the social coefficient, representing the particle's tendency to move toward the swarm's best-known position.  
  **Value**: `0.9`

- **Final Social Coefficient (`final_social`)**:  
  The final value for the social coefficient after dynamic adjustment.  
  **Value**: `0.4`

- **Social Adjustment Parameter (`social_param`)**:  
  The rate or factor influencing the change in the social coefficient.  
  **Value**: `0.85`
