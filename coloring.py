import math
import random
from typing import List, Set

N = 100
SHRINKING_FACTOR = 7.5
NO_PROGRESS_STREAK_THRESHOLD = 100
EPS = 0.00001


Partition = List[int]


class Vec:
    def __init__(self):
        self.v = [0.0 for _ in range(N)]

    def print(self, s: str):
        print(s, end=" ")
        for i in range(N):
            print(f"{self.v[i]:f}, ", end="")
        print()


class Mat:
    def __init__(self):
        self.m = [[0.0 for _ in range(N)] for _ in range(N)]

    def print(self, s: str):
        print(s, end=" ")
        for i in range(N):
            for j in range(N):
                print(f"{self.m[i][j]:f}, ", end="")
            print()
        print()

    def mult(self, v: Vec) -> Vec:
        r = Vec()
        for row in range(N):
            s = 0.0
            for col in range(N):
                s += self.m[row][col] * v.v[col]
            r.v[row] = s
        return r

    @staticmethod
    def gauss_seidel(x: Vec, b: Vec, m: "Mat", tol: float, maxiter: int, partitions: List[Partition]) -> int:
        iter_count = 0

        for iteration_index in range(maxiter):
            for partition in partitions:
                # Gauss-Seidel step for this partition.
                for variable in partition:
                    s = 0.0
                    for j in range(N):
                        if j != variable:
                            s += m.m[variable][j] * x.v[j]
                    x.v[variable] = (b.v[variable] - s) / m.m[variable][variable]

            mx = m.mult(x)

            norm = 0.0
            for i in range(N):
                a = mx.v[i] - b.v[i]
                norm += a * a
            norm = math.sqrt(norm)

            if norm < tol:
                iter_count = iteration_index + 1
                break
            iter_count = iteration_index + 1

        return iter_count


def randomized_graph_coloring(m: Mat) -> List[Partition]:
    neighbours: List[Set[int]] = [set() for _ in range(N)]

    node_colors = [0 for _ in range(N)]
    next_color = [0 for _ in range(N)]
    node_palettes: List[Set[int]] = [set() for _ in range(N)]
    U: Set[int] = set()

    # Build adjacency from non-zero off-diagonal matrix entries.
    for i in range(N):
        for j in range(N):
            if i != j and abs(m.m[i][j]) > EPS:
                neighbours[i].add(j)
                neighbours[j].add(i)

    # Maximum degree.
    delta_v = 0
    for i in range(N):
        if len(neighbours[i]) > delta_v:
            delta_v = len(neighbours[i])

    # Same behavior as the C++ code: compute, then override to 2.
    max_color = int(float(delta_v) / SHRINKING_FACTOR)
    if max_color <= 0:
        max_color = 1
    max_color = 2

    for iv in range(N):
        for ic in range(max_color):
            node_palettes[iv].add(ic)
        next_color[iv] = max_color

    for iv in range(N):
        U.add(iv)

    no_progress_streak = 0

    while len(U) > 0:
        # Assign random colors from each node's palette.
        for iv in U:
            palette_list = list(node_palettes[iv])
            node_colors[iv] = random.choice(palette_list)

        temp: Set[int] = set()

        for iv in U:
            icolor = node_colors[iv]

            different_from_neighbours = True
            for neighbour in neighbours[iv]:
                if node_colors[neighbour] == icolor:
                    different_from_neighbours = False
                    break

            if different_from_neighbours:
                # Remove this color from all neighbors' palettes.
                for neighbour in neighbours[iv]:
                    if icolor in node_palettes[neighbour]:
                        node_palettes[neighbour].remove(icolor)
            else:
                temp.add(iv)

            # Feed the hungry: if palette empty, add more colors on the fly.
            if len(node_palettes[iv]) == 0:
                node_palettes[iv].add(next_color[iv])
                next_color[iv] += 1

        if len(U) == len(temp):
            no_progress_streak += 1

            if no_progress_streak > NO_PROGRESS_STREAK_THRESHOLD:
                mnode = random.choice(list(U))
                node_palettes[mnode].add(next_color[mnode])
                next_color[mnode] += 1
                no_progress_streak = 0

        U = temp

    # Number of colors actually assigned to nodes.
    num_colors = max(node_colors) + 1 if node_colors else 0

    partitions: List[Partition] = []
    for ic in range(num_colors):
        partition: Partition = []
        for inode in range(N):
            if node_colors[inode] == ic:
                partition.append(inode)
        partitions.append(partition)

    return partitions


def get_diagonally_dominant_matrix() -> Mat:
    m = Mat()

    # Generate random matrix entries.
    for i in range(N):
        for j in range(i, N):
            if random.randrange(9) == 0:
                value = float(random.randrange(10))
                m.m[i][j] = value
                m.m[j][i] = value

    # Make each row diagonally dominant.
    for i in range(N):
        diag = abs(m.m[i][i])
        row_sum = 0.0

        for j in range(N):
            if i != j:
                row_sum += abs(m.m[i][j])

        if not (diag >= row_sum):
            m.m[i][i] += (row_sum - diag)

        if abs(m.m[i][i]) < EPS:
            m.m[i][i] += 1.0

    return m


def main():
    random.seed(13000)

    m = get_diagonally_dominant_matrix()

    non_zeros = 0
    for i in range(N):
        for j in range(N):
            if abs(m.m[i][j]) > EPS:
                non_zeros += 1

    print(f"percent of non-zeros of M: {int(100.0 * float(non_zeros) / float(N * N))}%")

    expected_solution = Vec()
    for i in range(N):
        expected_solution.v[i] = 8.0 * float(random.randrange(100)) / 100.0 - 4.0

    b = m.mult(expected_solution)

    x = Vec()  # initialized to all zeros

    print(f"solving linear system where N = {N}\n\n\n")

    expected_solution.print("expected solution:")
    print()

    partitions = randomized_graph_coloring(m)

    iter_count = Mat.gauss_seidel(x, b, m, 0.001, 10000, partitions)

    print(f"number of partitions: {len(partitions)}")
    x.print("gauss-seidel method solution:")
    print(f"number of iterations: {iter_count}")


if __name__ == "__main__":
    main()