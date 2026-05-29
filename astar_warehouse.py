"""
A* Search Algorithm – Warehouse Robot Navigation
=================================================

Navigates a robot from a start position to an end position on a 2-D binary
warehouse grid (1 = walkable, 0 = obstacle) using 8-directional movement.
Two heuristic functions are provided and compared:
  - manhattan_distance : L1 norm (sum of absolute row/col differences)
  - euclidean_distance : L2 norm (straight-line distance)
"""


import heapq
import math
import logging
import os
import glob
import re
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astar_log.txt")

# Clear log file at the start of each execution
open(log_file, "w").close()

with open(log_file, "a") as _f:
    _f.write("\n" + "=" * 60 + "\n")
    _f.write(f" NEW EXECUTION - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n")
    _f.write("=" * 60 + "\n\n")

# Root handler: write DEBUG+ to file only (no console noise)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(log_file, mode='a')],
)

logger = logging.getLogger(__name__)
logger.propagate = False        # Suppress propagation to root/console
logger.setLevel(logging.DEBUG)

# Detailed per-function formatter for step-by-step tracing
_file_handler = logging.FileHandler(log_file, mode='a')
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    fmt='%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
))

# Guard against duplicate handlers on re-import
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(_file_handler)


class Node:
    """
    Represents a single cell/node in the warehouse grid.

    Attributes
    ----------
    position : tuple (row, col)
        Grid coordinate of this node.
    parent : Node or None
        Reference to the parent node used for path reconstruction.
    g_cost : float
        Actual accumulated cost from the start node to this node.
    h_cost : float
        Heuristic estimated cost from this node to the goal.
    f_cost : float
        Total estimated cost: g_cost + h_cost.
    """

    def __init__(self, position, parent=None):
        self.position = position    # (row, col)
        self.parent   = parent      # Parent node for path tracing
        self.g_cost   = 0.0         # Actual cost from start
        self.h_cost   = 0.0         # Heuristic cost to goal
        self.f_cost   = 0.0         # Total estimated cost

    def __eq__(self, other):
        """Two nodes are equal when they occupy the same grid position."""
        return self.position == other.position

    def __lt__(self, other):
        """
        Heap comparison: lower f_cost has higher priority.
        Ties are broken in favour of the node with the lower h_cost,
        biasing expansion toward the goal.
        """
        if self.f_cost == other.f_cost:
            return self.h_cost < other.h_cost
        return self.f_cost < other.f_cost

    def __hash__(self):
        """Make Node hashable for use in sets and dicts."""
        return hash(self.position)

    def __repr__(self):
        return (
            f"Node(pos={self.position}, "
            f"g={self.g_cost:.4f}, h={self.h_cost:.4f}, f={self.f_cost:.4f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic Functions  
# ─────────────────────────────────────────────────────────────────────────────
def manhattan_distance(node_pos, goal_pos):
    """
    Manhattan distance heuristic (L1 norm).

    Admissible for 4-directional movement; underestimates on 8-directional
    grids, which may cause more nodes to be expanded.

    Formula:  h = |row_n - row_goal| + |col_n - col_goal|
    """
    x1, y1 = node_pos
    x2, y2 = goal_pos
    return abs(x1 - x2) + abs(y1 - y2)


def euclidean_distance(node_pos, goal_pos):
    """
    Euclidean distance heuristic (L2 norm).

    Admissible and consistent for 8-directional movement; accurately models
    the true straight-line distance and typically guides the search to
    expand fewer nodes when diagonal moves are weighted as sqrt(2).

    Formula:  h = sqrt((row_n - row_goal)^2 + (col_n - col_goal)^2)
    """
    x1, y1 = node_pos
    x2, y2 = goal_pos
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# A* Search Algorithm  
# ─────────────────────────────────────────────────────────────────────────────
def astar_search(warehouse, start, end, heuristic_func):
    """
    Perform A* pathfinding on a 2-D warehouse grid.

    Parameters
    ----------
    warehouse : list[list[int]]
        2-D grid where 1 = walkable cell and 0 = obstacle.
    start : tuple (row, col)
        Starting position of the robot.
    end : tuple (row, col)
        Goal / delivery position.
    heuristic_func : callable(pos, goal) -> float
        Heuristic that estimates the remaining cost from *pos* to *goal*.

    Returns
    -------
    path : list[tuple] or None
        Ordered list of (row, col) positions from start to end (inclusive),
        or None if no path exists.
    cost : float
        Total accumulated path cost, or 0 if no path was found.
    """
    logger.debug(
        f"Starting A* search | Start: {start} | End: {end} | "
        f"Heuristic: {heuristic_func.__name__}"
    )

    # ── Grid Validation ───────────────────────────────────────────────────────
    if not warehouse or not warehouse[0]:
        logger.error("Warehouse grid is empty or invalid.")
        return None, 0

    rows = len(warehouse)
    cols = len(warehouse[0])
    logger.debug(f"Grid dimensions: {rows} rows x {cols} cols")

    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        logger.error(f"Start position {start} is out of grid bounds.")
        return None, 0

    if not (0 <= end[0] < rows and 0 <= end[1] < cols):
        logger.error(f"End position {end} is out of grid bounds.")
        return None, 0

    if warehouse[start[0]][start[1]] == 0:
        logger.error(f"Start position {start} is blocked by an obstacle.")
        return None, 0

    if warehouse[end[0]][end[1]] == 0:
        logger.error(f"End position {end} is blocked by an obstacle.")
        return None, 0

    # ── Initialise Start and Goal Nodes ───────────────────────────────────────
    start_node         = Node(start)
    start_node.g_cost  = 0.0
    start_node.h_cost  = heuristic_func(start, end)
    start_node.f_cost  = start_node.h_cost

    end_node = Node(end)

    # ── Open List (min-heap) and Supporting Structures ────────────────────────
    open_list = []
    heapq.heappush(open_list, start_node)

    # Best known g_cost to each visited position
    g_score = {start: 0.0}

    # Fast O(1) membership check to prevent duplicate heap pushes
    open_set = {start}

    # ── Movement Directions ───────────────────────────────────────────────────
    # (row_delta, col_delta, movement_cost)
    # Cardinal moves cost 1; diagonal moves cost sqrt(2) ≈ 1.414
    movements = [
        ( 0,  1, 1              ),   # Right   (East)
        ( 0, -1, 1              ),   # Left    (West)
        ( 1,  0, 1              ),   # Down    (South)
        (-1,  0, 1              ),   # Up      (North)
        ( 1,  1, math.sqrt(2)  ),   # Down-Right  (South-East)
        ( 1, -1, math.sqrt(2)  ),   # Down-Left   (South-West)
        (-1,  1, math.sqrt(2)  ),   # Up-Right    (North-East)
        (-1, -1, math.sqrt(2)  ),   # Up-Left     (North-West)
    ]

    iterations = 0

    # ── Main A* Loop ──────────────────────────────────────────────────────────
    while open_list:
        iterations += 1

        current_node = heapq.heappop(open_list)
        open_set.discard(current_node.position)

        logger.debug(
            f"Iter {iterations:04d} | Exploring {current_node.position} | "
            f"g={current_node.g_cost:.4f}, h={current_node.h_cost:.4f}, "
            f"f={current_node.f_cost:.4f}"
        )

        # ── Goal Check ────────────────────────────────────────────────────────
        if current_node == end_node:
            path = []
            node = current_node
            while node is not None:
                path.append(node.position)
                node = node.parent
            path.reverse()          # Order: start → end

            logger.info(
                f"Path found! | Iterations: {iterations} | "
                f"Path length: {len(path)} steps | "
                f"Total cost: {current_node.g_cost:.4f}"
            )
            logger.debug(f"Full path: {path}")
            return path, current_node.g_cost

        # ── Expand Neighbours ─────────────────────────────────────────────────
        for dr, dc, move_cost in movements:
            neighbor_pos = (
                current_node.position[0] + dr,
                current_node.position[1] + dc,
            )

            # Boundary check
            if not (0 <= neighbor_pos[0] < rows and 0 <= neighbor_pos[1] < cols):
                continue

            # Obstacle check
            if warehouse[neighbor_pos[0]][neighbor_pos[1]] == 0:
                logger.debug(f"  Obstacle at {neighbor_pos} — skipped")
                continue

            tentative_g = (
                g_score.get(current_node.position, float('inf')) + move_cost
            )

            # Only update when a strictly better path is found
            if tentative_g < g_score.get(neighbor_pos, float('inf')):
                neighbor_node          = Node(neighbor_pos, parent=current_node)
                neighbor_node.g_cost   = tentative_g
                neighbor_node.h_cost   = heuristic_func(neighbor_pos, end)
                neighbor_node.f_cost   = neighbor_node.g_cost + neighbor_node.h_cost

                logger.debug(
                    f"  Update {neighbor_pos} | "
                    f"g={neighbor_node.g_cost:.4f}, h={neighbor_node.h_cost:.4f}, "
                    f"f={neighbor_node.f_cost:.4f}"
                )

                g_score[neighbor_pos] = tentative_g

                if neighbor_pos not in open_set:
                    heapq.heappush(open_list, neighbor_node)
                    open_set.add(neighbor_pos)

    # ── No Path Found ─────────────────────────────────────────────────────────
    logger.warning(
        f"No path found from {start} to {end} after {iterations} iterations."
    )
    return None, 0


# ─────────────────────────────────────────────────────────────────────────────
# Grid Visualisation & Test Case Execution  
# ─────────────────────────────────────────────────────────────────────────────
def plot_path(warehouse, path, start, end, title):
    """
    Render the warehouse grid with the computed path overlaid.

    Parameters
    ----------
    warehouse : numpy.ndarray
        2-D binary grid (1 = free, 0 = obstacle).
    path : list[tuple] or None
        Sequence of (row, col) positions representing the route.
    start : tuple (row, col)
        Start cell (rendered green, labelled 'S').
    end : tuple (row, col)
        Goal cell (rendered red, labelled 'E').
    title : str
        Figure title.
    """
    rows, cols = warehouse.shape
    grid = np.array(warehouse)

    fig, ax = plt.subplots(figsize=(cols, rows))

    for r in range(rows):
        for c in range(cols):
            color = 'lightgray' if grid[r, c] == 0 else 'white'
            ax.add_patch(mpatches.Rectangle((c, r), 1, 1,
                                            facecolor=color, edgecolor='black'))

    # Start cell
    ax.add_patch(mpatches.Rectangle(
        (start[1], start[0]), 1, 1, facecolor='green', edgecolor='black', label='Start'))
    ax.text(start[1] + 0.5, start[0] + 0.5, 'S',
            ha='center', va='center', color='white', fontweight='bold')

    # End cell
    ax.add_patch(mpatches.Rectangle(
        (end[1], end[0]), 1, 1, facecolor='red', edgecolor='black', label='End'))
    ax.text(end[1] + 0.5, end[0] + 0.5, 'E',
            ha='center', va='center', color='white', fontweight='bold')

    # Path overlay
    if path:
        path_x = [p[1] + 0.5 for p in path]
        path_y = [p[0] + 0.5 for p in path]
        ax.plot(path_x, path_y, color='blue', linewidth=2,
                marker='o', markersize=5, label='Path')

    ax.set_xticks(np.arange(cols + 1))
    ax.set_yticks(np.arange(rows + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)          # Invert y-axis: (0,0) at top-left
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.grid(False)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    input_files  = glob.glob(os.path.join(script_dir, "inputPS*.txt")
                             ) or glob.glob(os.path.join(script_dir, "InputPS*.txt"))
    output_file  = os.path.join(script_dir, "outputPS05.txt")

    if not input_files:
        print("No input file starting with 'inputPS' found in the script directory.")
    else:
        all_test_cases = []   # accumulate across all input files

        for input_file in input_files:
            print(f"\nReading input from: {input_file}")
            with open(input_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]

            file_test_cases = []
            for line in lines:
                if 'warehouse' in line and 'start' in line and 'end' in line:
                    warehouse_match = re.search(
                        r'warehouse\s*=\s*(\[.*?\](?:\s*,\s*\[.*?\])*\])', line)
                    start_match = re.search(
                        r'start\s*=\s*\((\d+),\s*(\d+)\)', line)
                    end_match = re.search(
                        r'end\s*=\s*\((\d+),\s*(\d+)\)', line)

                    if warehouse_match and start_match and end_match:
                        warehouse_data = eval(
                            warehouse_match.group(0).split('=')[1].strip())
                        start_data = (
                            int(start_match.group(1)),
                            int(start_match.group(2)),
                        )
                        end_data = (
                            int(end_match.group(1)),
                            int(end_match.group(2)),
                        )
                        file_test_cases.append((warehouse_data, start_data, end_data))

            if not file_test_cases:
                print("  No valid test cases found in this file.")
                continue

            print(f"  Found {len(file_test_cases)} test case(s).")
            for idx, (wh, s, e) in enumerate(file_test_cases, 1):
                print(f"\n--- Test Case {idx} ---")
                print(f"  Grid ({len(wh)}x{len(wh[0])}), Start: {s}, End: {e}")
                print("  Warehouse Matrix:")
                for row in wh:
                    print(f"    {row}")

                path_m, cost_m = astar_search(wh, s, e, manhattan_distance)
                path_e, cost_e = astar_search(wh, s, e, euclidean_distance)

                print(f"  Manhattan Path : {path_m}")
                print(f"  Manhattan Cost : {cost_m:.2f}")
                print(f"  Euclidean Path : {path_e}")
                print(f"  Euclidean Cost : {cost_e:.2f}")

            all_test_cases.extend(file_test_cases)

        # ── Write output file ──────────────────────────────────────────────
        if all_test_cases:
            with open(output_file, "w") as f:
                for i, (warehouse, start, end) in enumerate(all_test_cases, 1):
                    path_m, cost_m = astar_search(warehouse, start, end, manhattan_distance)
                    path_e, cost_e = astar_search(warehouse, start, end, euclidean_distance)

                    # Pick the cheaper path; prefer Euclidean on a tie
                    if cost_e <= cost_m:
                        best_heuristic = "Euclidean"
                        best_cost      = cost_e
                        best_path      = path_e
                    else:
                        best_heuristic = "Manhattan"
                        best_cost      = cost_m
                        best_path      = path_m

                    f.write(f"Test Case {i}\n")
                    f.write("Path:\n")

                    if best_path:
                        if len(best_path) == 1:
                            # Single-node path: start == end
                            f.write(f"  [{best_path[0]}]\n")
                        else:
                            for j, node in enumerate(best_path):
                                if j == 0:
                                    f.write(f"  [{node},\n")
                                elif j == len(best_path) - 1:
                                    f.write(f"  {node}]\n")
                                else:
                                    f.write(f"  {node},\n")
                    else:
                        f.write("  No path found\n")

                    f.write(f"Heuristic: {best_heuristic}\n")
                    f.write(f"Cost: {best_cost:.2f}\n")
                    f.write("\n")

            print(f"\nOutput written to: {output_file}")
