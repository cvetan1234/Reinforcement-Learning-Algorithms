import random
import numpy as np
import matplotlib.pyplot as plt

from gridworld import GridworldEnv
from plot import plot_v_table

class MazeDFS(GridworldEnv):
    """
    A maze environment generated using randomized Depth-First Search (DFS),
    extending the GridworldEnv class.

    The maze is guaranteed to be solvable, with a start point ('S') and
    a goal point ('G'). Walls are represented with '#', and free paths
    with spaces (' ').

    Attributes:
        step_cost (int): The penalty for each step taken in the maze.
        g_reward (int): The reward for reaching the goal.
        o_reward (int): The penalty for invalid or obstacle moves.
        seed (int or None): Random seed for maze reproducibility.
        size (int): The size of the maze (will be made odd if even).
        grid (List[List[str]]): 2D grid representation of the maze.
    """

    def __init__(self, size=11, seed=None):
        """
        Initialize the maze with a given size and optional seed.

        Args:
            size (int): Desired size of the maze (default is 11). Adjusted to be odd.
            seed (int, optional): Seed for random number generation (for reproducible mazes).
        """
        self.step_cost = -1
        self.g_reward = 100
        self.o_reward = -100
        self.seed = seed

        # Ensure maze size is odd to allow perfect maze generation
        if size % 2 == 0:
            size += 1
        self.size = size
        self.grid = [['#' for _ in range(size)] for _ in range(size)]   # Initialize the grid with walls ('#')
        self._generate_maze()
        super().__init__()

    def _generate_maze(self):
        """
        Generate the maze layout using recursive randomized DFS.
        Ensures all cells are reachable and the maze is solvable.

        The method carves paths from the initial point (1,1), creating
        corridors by removing walls between cells in random directions.
        """
        # Set the random seed (to ensure the same maze is generated every time)
        if self.seed is not None:
            random.seed(self.seed)

        visited = [[False for _ in range(self.size)] for _ in range(self.size)]

        def carve(x, y):
            """
            Recursive function to carve paths in the maze grid.

            Args:
                x (int): Current x-coordinate in the grid.
                y (int): Current y-coordinate in the grid.
            """
            dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            random.shuffle(dirs)
            visited[x][y] = True
            self.grid[x][y] = ' '

            for dx, dy in dirs:    # Try each direction
                nx, ny = x + dx, y + dy
                # If the target cell is within bounds and unvisited, carve to it
                if 0 <= nx < self.size and 0 <= ny < self.size and not visited[nx][ny]:
                    self.grid[x + dx // 2][y + dy // 2] = ' '
                    carve(nx, ny)

        carve(1, 1)
        self.grid[1][1] = 'S'
        self.grid[self.size - 2][self.size - 2] = 'G'

def visualize_maze(env):
    """
    Visualize the MazeDFS environment using matplotlib.

    Converts the maze grid to a grayscale image where:
    - Free paths are white
    - Walls are black

    Args:
        env (MazeDFS): An instance of the MazeDFS environment to visualize.
    """
    m, n = env.shape()
    maze_array = np.ones((m, n))

    # Convert grid characters to numeric values for visualization
    for x in range(m):
        for y in range(n):
            if env.grid[x][y] == '#':
                maze_array[x, y] = 0    # Wall is black (0)

    # Display the maze
    plt.figure(figsize=(n / 2, m / 2))
    plt.imshow(maze_array, cmap='gray', origin='upper')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    seed = 100
    env = MazeDFS(size=25, seed=seed)

    visualize_maze(env)

