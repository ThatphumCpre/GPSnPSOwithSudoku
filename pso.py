import numpy as np
from typing import List, Optional
import random
import logging
import os 
import sys

from sudoku import load_board

class SudokuPuzzle:
    logger = logging.getLogger(__name__)
    def __init__(self, grid: list[list[int]]):
        self.grid = np.array(grid)

    def is_valid(self) -> bool:
        for row in range(9):
            if not self.is_valid_line(self.grid[row, :]):
                return False
        for col in range(9):
            if not self.is_valid_line(self.grid[:, col]):
                return False
        for box_row in range(3):
            for box_col in range(3):
                if not self.is_valid_box(box_row, box_col):
                    return False
        return True

    def is_valid_line(self, line: np.ndarray) -> int :
        elements = line[line != 0]
        return len(np.unique(elements))

    def is_valid_box(self, box_row: int, box_col: int) -> bool:
        box = self.grid[
            box_row * 3 : (box_row + 1) * 3, box_col * 3 : (box_col + 1) * 3
        ].flatten()
        elements = box[box != 0]
        return len(np.unique(elements))

    def __str__(self) -> str:
        return "\n".join([" ".join(map(str, row)) for row in self.grid])



from IPython.display import clear_output
class Particle:
    def __init__(self, sudoku_puzzle: SudokuPuzzle):
        self.fixed_positions = sudoku_puzzle.grid != 0
        self.position = self.initialize_position(sudoku_puzzle)
        self.velocity = np.random.randint(1, 10, size=(9, 9))
        self.best_position = self.position.copy()
        self.best_score = self.evaluate(sudoku_puzzle)

    def initialize_position(self, sudoku_puzzle: SudokuPuzzle) -> np.ndarray:
        position = sudoku_puzzle.grid.copy()
        for row in range(9):
            for col in range(9):
                if position[row, col] == 0:
                    position[row, col] = random.randint(1, 9)
        return position

    def evaluate(self, sudoku_puzzle: SudokuPuzzle) -> int:
        temp_grid = sudoku_puzzle.grid.copy()
        sudoku_puzzle.grid = self.position.copy()

        row_score = sum(
            sudoku_puzzle.is_valid_line(sudoku_puzzle.grid[row, :])
            for row in range(9)
        )
        col_score = sum(
            sudoku_puzzle.is_valid_line(sudoku_puzzle.grid[:, col])
            for col in range(9)
        )
        box_score = sum(
            sudoku_puzzle.is_valid_box(box_row, box_col)
            for box_row in range(3)
            for box_col in range(3)
        )
        sudoku_puzzle.grid = temp_grid
        total_score = row_score + col_score + box_score

        return total_score

    def update_velocity(self, global_best_position: np.ndarray, w=0.5, c1=1, c2=1):
        r1 = np.random.rand(9, 9)
        r2 = np.random.rand(9, 9)
        cognitive_velocity = c1 * r1 * (self.best_position - self.position)
        social_velocity = c2 * r2 * (global_best_position - self.position)
        next_velocity = w * self.velocity + cognitive_velocity + social_velocity


        self.velocity = next_velocity

    def update_position(self):
        new_position = self.position + self.velocity.astype(int)
        new_position = (new_position - 0) % 10


        self.position = np.where(self.fixed_positions, self.position, new_position)


def str2d(p):
    puzzlestring = ""
    for i in range(9):
        puzzle = [p[i][j] for j in range(9)]
        puzzlestring += "{p[0]} {p[1]} {p[2]} | {p[3]} {p[4]} {p[5]} | {p[6]} {p[7]} {p[8]}\n".format(p=puzzle)
        if i in [2,5]:
            puzzlestring += "----------------------\n"
    return puzzlestring.replace('None',' ')

def str2d_highight(p):
    puzzlestring = ""
    # Track value frequencies
    frequencies = {}
    for i in range(9):
        row = [int(round(p[i][j])) for j in range(9)]
        # Count frequencies of non-None values in the row
        row_freq = {}
        for val in row:
            if val is not None:
                row_freq[val] = row_freq.get(val, 0) + 1
        
        # Prepare row for formatting with color highlighting
        formatted_row = []
        for j in range(9):
            val = row[j]
            if val is None:
                formatted_row.append(' ')
            else:
                # Check if the value appears more than once in the row
                is_non_unique = row_freq.get(val, 0) > 1
                
                # Prepare colored formatting
                if is_non_unique:
                    formatted_row.append(f'\033[91m{val}\033[0m')  # Red color
                else:
                    formatted_row.append(str(val))
        
        # Format the row with separators
        puzzlestring += "{p[0]} {p[1]} {p[2]} | {p[3]} {p[4]} {p[5]} | {p[6]} {p[7]} {p[8]}\n".format(p=formatted_row)
        
        # Add separator lines after specific rows
        if i in [2, 5]:
            puzzlestring += "----------------------\n"
    
    return puzzlestring


class Swarm:
    def __init__(self, sudoku_puzzle: SudokuPuzzle, num_particles: int):
        self.particles = [Particle(sudoku_puzzle) for _ in range(num_particles)]
        self.global_best_position = self.particles[0].position.copy()
        self.global_best_score = self.particles[0].evaluate(sudoku_puzzle)

    def update_global_best(self, sudoku_puzzle: SudokuPuzzle):
        for particle in self.particles:
            score = particle.evaluate(sudoku_puzzle)
    
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = particle.position.copy()


import matplotlib.pyplot as plt

class PsoSolver:
    def __init__(self, sudoku_puzzle: SudokuPuzzle, num_particles=30, max_iter=100):
        self.sudoku_puzzle = sudoku_puzzle
        self.swarm = Swarm(sudoku_puzzle, num_particles)
        self.max_iter = max_iter

    def solve(self) -> SudokuPuzzle:
        global_best_scores = []  # List to track global best scores

        for iteration in range(self.max_iter):
            clear_output(wait=True)
            os.system('cls')
            for particle in self.swarm.particles:
                particle.update_velocity(self.swarm.global_best_position)
                particle.update_position()
                score = particle.evaluate(self.sudoku_puzzle)

                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()

            self.swarm.update_global_best(self.sudoku_puzzle)
            
            # Store global best score
            global_best_scores.append(self.swarm.global_best_score)

            # Print progress for each iteration
            print("PsoSolver.solve(): start")
            print(f"Iteration {iteration + 1}/{self.max_iter}, Best Score: {self.swarm.global_best_score}")
            print(f"Global best position\n{str2d(self.swarm.global_best_position.tolist())}\n")
            print("PsoSolver.solve(): end")

            if self.swarm.global_best_score == 243:  # Sudoku solved
                print("Sudoku solved!: self.swarm.global_best_score == 243")
                break

        # Plot the global best scores
        plt.figure(figsize=(10, 6))
        plt.plot(global_best_scores, marker='o', label='Global Best Score')
        plt.title('PSO Global Best Score by Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Global Best Score')
        plt.grid(True)
        plt.legend()
        plt.show()

        self.sudoku_puzzle.grid = self.swarm.global_best_position
        return self.sudoku_puzzle
    

def main(puzzle):
    # Example Sudoku puzzle (0 represents empty cells)


    # Initialize Sudoku puzzle
    sudoku_puzzle = SudokuPuzzle(puzzle)

    # Print the initial puzzle
    print("Initial Sudoku Puzzle:")
    print(sudoku_puzzle)

    # Initialize PSO solver
    pso_solver = PsoSolver(sudoku_puzzle, num_particles=1000, max_iter=1000)

    # Solve the puzzle
    solved_puzzle = pso_solver.solve()

    # Print the solved puzzle
    print("\nThe problems we attacked :")
    converted_puzzle = [['.' if cell == 0 else cell for cell in row] for row in puzzle]
    print(str2d(converted_puzzle))
    print("\nSolved Sudoku Puzzle:")
    print(str2d_highight(solved_puzzle.grid.tolist()))


if __name__ == "__main__":
    print(sys.argv)
    puzzle = load_board(sys.argv[1])[0]
    print(puzzle)
    main(puzzle)