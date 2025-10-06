# demo.py
# Copyright (c) 2025 Tobias Karusseit
# Licensed under the MIT License. See LICENSE file in the project root for full license information

import random
import pandas as pd


# simple cell class for the maze
class Cell:
    def __init__(self, x, y):
        self.id = f"cell-{x, y}"
        self.x = x
        self.y = y
        self.visited = False
        self.connected = []


def main():
    # init a 10x10 maze
    mazeHeight = 10
    mazeWidth = 10
    # init the cells
    cells = [Cell(x, y) for x, y in [(x, y) for x in range(mazeWidth) for y in range(mazeHeight)]]
    cellStack = []
    # random choose one cell and init the stack
    random.shuffle(cells)
    cellStack.append(cells.pop())
    # build the maze
    while cellStack:
        currentCell = cellStack[-1] # peek at the current cell
        # get all not et visited neighbors
        neighbors = [
            c for c in cells if not c.visited and (
                (c.x == currentCell.x + 1 and c.y == currentCell.y) or
                (c.x == currentCell.x - 1 and c.y == currentCell.y) or
                (c.y == currentCell.y + 1 and c.x == currentCell.x) or
                (c.y == currentCell.y - 1 and c.x == currentCell.x)
            )
        ]

        if neighbors:
            # select a random neighbor
            random.shuffle(neighbors)
            neighbor = neighbors.pop()
            # connect the cells (remove walls)
            currentCell.connected.append(neighbor)
            neighbor.connected.append(currentCell)
            # mark the neighbor as visited
            neighbor.visited = True
            # add the neighbor to the stack
            cellStack.append(neighbor)
        else:
            # remove the current cell from the stack since no paths exist for it
            cellStack.pop()

    # init the dataframe
    data = pd.DataFrame(columns=[
        "source",
        "destination",
        "distance",
        "source_lat",
        "source_lng",
        "destination_lat",
        "destination_lng"
    ])
    # add the edges to the dataframe
    for cell in cells:
        for neighbor in cell.connected:
            data.loc[len(data)] = [cell.id, neighbor.id, 1, cell.x, cell.y, neighbor.x, neighbor.y]
    # save the dataframe
    data.to_csv("docs/examples/mazePathfinder/data/maze.csv", index=False)


if __name__ == "__main__":
    main()
