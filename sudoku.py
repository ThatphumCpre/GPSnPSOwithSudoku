import random
import numpy
import copy
import matplotlib.pyplot as plt

# Configuration
swarm_size = 100 
mutation_prob = 0.5
max_iterations = 1000
# Dynamic Coefficient
dynamic_parameter = True
# Cognitive Coefficient(c1)
initial_cognitive = 0.1
final_cognitive = 0.5
cognitive_param = 0.15
# Social Coefficient(c2)
initial_social = 0.9
final_social = 0.4
social_param = 0.85

# Load the Sudoku board from a file
def load_board(file_name):
    file = open(file_name, 'r')
    board = [[0] * 9 for _ in range(9)]
    i = 0
    for line in file:
        board[i] = line.split()
        i += 1
        if i == 9:
            break
    unresolved_indices = []
    for i in range(9):
        indices = []
        for j in range(9):
            board[i][j] = int(board[i][j])
            if board[i][j] == 0:
                indices.append(j)
        unresolved_indices.append(indices)

    file.close()
    return board, unresolved_indices


# Initialize the swarm of particles
def initialize_swarm(swarm_size, board):
    swarm = []
    for _ in range(swarm_size):
        particle = copy.deepcopy(board)
        for i in range(9):
            permutation = numpy.random.permutation([1, 2, 3, 4, 5, 6, 7, 8, 9])
            j = 0
            for number in permutation:
                if number in particle[i]:
                    continue
                else:
                    while particle[i][j] != 0:
                        j += 1
                    particle[i][j] = number
        swarm.append(particle)

    return swarm

# Fitness function to evaluate the swarm
def fitness_function(swarm):
    fitness_values = []
    for particle in swarm:
        fitness = 0
        for i in range(9):
            fitness += len(set(particle[i]))

        for i in range(9):
            fitness += len(set([col[i] for col in particle]))

        for i in range(3):
            for j in range(3):
                square = [particle[y][x] for y in range(3 * i, 3 * i + 3) for x in range(3 * j, 3 * j + 3)]
                fitness += len(set(square))

        fitness_values.append(fitness)

    return fitness_values

# Swap values in a row
def swap(row, value, swap_index):
    swap_index_2 = row.index(value)
    row[swap_index], row[swap_index_2] = row[swap_index_2], row[swap_index]
    return row

# Convex combination for particle updating
def convex_combination(swarm, nlc, glc, cognitive_param, social_param, swarm_size):
    new_swarm = copy.deepcopy(swarm)
    new_nlc = copy.deepcopy(nlc)
    new_glc = copy.deepcopy(glc)

    for r in range(swarm_size):
        for i in range(9):
            mask = [0] * 9
            for j in range(9):  # Generate random mask
                draw = random.random()
                if draw < social_param:
                    mask[j] = 3
                elif draw >= (1 - cognitive_param):
                    mask[j] = 2
                else:
                    mask[j] = 1

            for swap_index in range(9):
                if mask[swap_index] == 1:
                    value = new_swarm[r][i][swap_index]
                    new_nlc[r][i] = swap(new_nlc[r][i], value, swap_index)
                    new_glc[i] = swap(new_glc[i], value, swap_index)
                elif mask[swap_index] == 2:
                    value = new_nlc[r][i][swap_index]
                    new_swarm[r][i] = swap(new_swarm[r][i], value, swap_index)
                    new_glc[i] = swap(new_glc[i], value, swap_index)
                elif mask[swap_index] == 3:
                    value = new_glc[i][swap_index]
                    new_swarm[r][i] = swap(new_swarm[r][i], value, swap_index)
                    new_nlc[r][i] = swap(new_nlc[r][i], value, swap_index)
    return new_swarm

# Mutation function to modify particles
def mutate(swarm, unresolved_indices, mutation_prob):
    for particle in swarm:
        for i in range(9):
            if random.random() < mutation_prob:
                index1, index2 = random.sample(unresolved_indices[i], 2)
                particle[i][index1], particle[i][index2] = particle[i][index2], particle[i][index1]

    return swarm

# Update the local best particles
def update_nlc(swarm, nlc, nlc_fitness, swarm_size):
    fitness_values = fitness_function(swarm)
    for i in range(swarm_size):
        if fitness_values[i] >= nlc_fitness[i]:
            nlc[i] = swarm[i]
            nlc_fitness[i] = fitness_values[i]

    return nlc, nlc_fitness

# Update the global best particle
def update_glc(nlc, nlc_fitness):
    best_index = int(numpy.argmax(nlc_fitness))
    glc = nlc[best_index]
    glc_fitness = nlc_fitness[best_index]

    return glc, glc_fitness

# Final mutation to resolve duplicates
def final_mutation(glc_):
    duplicate_indices = []
    glc = copy.deepcopy(glc_)

    # Process duplicates
    for i in range(9):
        column = [col[i] for col in glc]
        if len(set(column)) != 9:
            duplicate = set([x for x in column if column.count(x) > 1])
            for j in range(9):
                if glc[j][i] == list(duplicate)[0]:
                    duplicate_index = [j, i]
                    duplicate_indices.append(duplicate_index)

    # Further resolve duplicates
    if fitness_function([glc])[0] != 243:
        return glc_
    else:
        return glc

def print_board(board):
    for i, row in enumerate(board):
        # Print horizontal dividers every 3 rows
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        row_str = ""
        for j, value in enumerate(row):
            # Add vertical dividers every 3 columns
            if j % 3 == 0 and j != 0:
                row_str += "| "
            row_str += f"{value if value != 0 else '.'} "  # Use '.' for empty cells
        print(row_str)
    print("\n")  # Add extra space for readability

def print_wrong_board(board, initial_board):

    def find_incorrect_cells(board,initial_board):
        incorrect_cells = set()
        # Check rows for duplicates
        for i in range(9):
            seen = set()
            for j in range(9):
                value = initial_board[i][j]
                seen.add(value)
            for j in range(9):
                value = board[i][j]
                if value != 0 and value in seen and initial_board[i][j] == 0:
                    incorrect_cells.add((i, j))
                seen.add(value)
        # Check columns for duplicates
        for j in range(9):
            seen = set()
            for i in range(9):
                value = initial_board[i][j]
                seen.add(value)
            for i in range(9):
                value = board[i][j]
                if value != 0 and value in seen and initial_board[i][j] == 0:
                    incorrect_cells.add((i, j))
                seen.add(value)
        # Check 3x3 subgrids for duplicates
        for grid_row in range(3):
            for grid_col in range(3):
                seen = set()
                for i in range(3 * grid_row, 3 * grid_row + 3):
                    for j in range(3 * grid_col, 3 * grid_col + 3):
                        value = initial_board[i][j]
                        seen.add(value)
                for i in range(3 * grid_row, 3 * grid_row + 3):
                    for j in range(3 * grid_col, 3 * grid_col + 3):
                        value = board[i][j]
                        if value != 0 and value in seen and initial_board[i][j] == 0:
                            incorrect_cells.add((i, j))
                        seen.add(value)
        return incorrect_cells

    # Identify incorrect cells
    incorrect_cells = find_incorrect_cells(board,initial_board)

    # Print the board with highlighting
    for i, row in enumerate(board):
        # Print horizontal dividers every 3 rows
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        row_str = ""
        for j, value in enumerate(row):
            # Add vertical dividers every 3 columns
            if j % 3 == 0 and j != 0:
                row_str += "| "
            # Highlight incorrect cells, but only if they are not part of the initial board
            if (i, j) in incorrect_cells:
                row_str += f"\033[91m{value}\033[0m "
            else:
                row_str += f"{value if value != 0 else '.'} "
        print(row_str)
    print("\n")


if __name__ == '__main__':
    # Input parameters for the PSO Sudoku solver
    board_file_name = input("Enter the file name of the Sudoku board: ")
    print("\n")

    inertia_param = (1 - social_param - cognitive_param)  # Inertia parameter
    # Algorithm initialization
    best_fitness = []
    iteration_vector = []

    board, unresolved_indices = load_board(board_file_name)
    initial_board = board
    swarm = initialize_swarm(swarm_size, board)
    nlc = copy.deepcopy(swarm)
    nlc_fitness = fitness_function(nlc)
    glc, glc_fitness = update_glc(nlc, nlc_fitness)

    for iteration in range(max_iterations):
        # Dynamically adjust parameters
        if dynamic_parameter:
            social_param = initial_social - (initial_social - final_social) * (iteration / max_iterations)
            cognitive_param = initial_cognitive + (final_cognitive - initial_cognitive) * (iteration / max_iterations)
        else:
            social_param = social_param
            cognitive_param = cognitive_param

        swarm = convex_combination(swarm, nlc, glc, cognitive_param, social_param, swarm_size)
        swarm = mutate(swarm, unresolved_indices, mutation_prob)
        nlc, nlc_fitness = update_nlc(swarm, nlc, nlc_fitness, swarm_size)
        glc, glc_fitness = update_glc(nlc, nlc_fitness)

        print(f"Iteration {iteration + 1} | Best fitness: {glc_fitness}")
        best_fitness.append(glc_fitness)
        iteration_vector.append(iteration + 1)

        if glc_fitness == 243:
            break
    print()
    print("Initial Board : ")
    print_board(initial_board)
    if fitness_function([glc])[0] != 243:
        print("No complete solution could be found:")
        print_wrong_board(glc,initial_board)
    else:
        print("Solution found:")
        print_board(glc)

    plt.figure(1)
    plt.plot(iteration_vector, best_fitness, 'k-')
    plt.xlabel('Algorithm Iterations')
    plt.ylabel('Best Particle Fitness')
    plt.show()
