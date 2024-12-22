from typing import List, Optional

from .constants import Constants

DIRECTIONS = Constants.DIRECTIONS
RESOURCE_TYPES = Constants.RESOURCE_TYPES


class Resource:
    """Represents a resource tile in the game world.

    Resources are collectible items on the map that units can gather:
    - Wood: Basic resource, available from start
    - Coal: Requires research to collect
    - Uranium: Requires advanced research to collect

    Args:
        r_type (str): Resource type (WOOD, COAL, URANIUM)
        amount (int): Quantity of resource remaining

    Note:
        Resource types have different collection rates and
        fuel values when used in cities.
    """
    def __init__(self, r_type: str, amount: int):
        self.type = r_type
        self.amount = amount


class Cell:
    """Represents a single tile in the game map.

    Each cell can contain:
    - A resource (wood, coal, uranium)
    - A city tile
    - A road (with varying levels)
    
    The cell system forms the basis of the spatial
    representation used in the observation space.

    Args:
        x (int): X-coordinate on map
        y (int): Y-coordinate on map

    Attributes:
        pos (Position): Position object for location
        resource (Optional[Resource]): Resource on this tile if any
        citytile (Optional[CityTile]): City tile if present
        road (float): Road level (0.0-1.0, higher = faster movement)
    """
    def __init__(self, x, y):
        self.pos = Position(x, y)
        self.resource: Optional[Resource] = None
        self.citytile = None
        self.road = 0

    def has_resource(self):
        """Checks if this cell has available resources.

        Returns:
            bool: True if cell has resources with amount > 0
        """
        return self.resource is not None and self.resource.amount > 0


class GameMap:
    """2D grid representation of the game world.

    The GameMap provides the spatial component of the game state,
    organizing cells in a 2D grid. This forms a crucial part of
    the observation space for the RL agent, encoding:
    - Resource locations and amounts
    - City tile positions
    - Road levels
    - Unit positions (via queries)

    Args:
        width (int): Map width in cells
        height (int): Map height in cells

    Implementation Details:
    - Uses row-major order for cell storage
    - Provides efficient position-based lookups
    - Maintains game state spatial information
    - Core component of observation encoding
    """
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.map: List[List[Cell]] = [None] * height
        for y in range(0, self.height):
            self.map[y] = [None] * width
            for x in range(0, self.width):
                self.map[y][x] = Cell(x, y)

    def get_cell_by_pos(self, pos) -> Cell:
        """Retrieves cell at a given Position.

        Args:
            pos (Position): Position to query

        Returns:
            Cell: Cell at the specified position
        """
        return self.map[pos.y][pos.x]

    def get_cell(self, x, y) -> Cell:
        """Retrieves cell at given coordinates.

        Args:
            x (int): X-coordinate
            y (int): Y-coordinate

        Returns:
            Cell: Cell at the specified coordinates
        """
        return self.map[y][x]

    def _setResource(self, r_type, x, y, amount):
        """Internal method to update resource state.

        Used by the game engine to update resource amounts.
        Should not be called directly by agents or the RL
        environment.

        Args:
            r_type (str): Resource type (WOOD, COAL, URANIUM)
            x (int): X-coordinate
            y (int): Y-coordinate
            amount (int): New resource amount
        """
        cell = self.get_cell(x, y)
        cell.resource = Resource(r_type, amount)


class Position:
    """Represents a location in the game world.

    Provides utilities for:
    - Distance calculations (Manhattan distance)
    - Direction computation
    - Position translation
    - Adjacency checks

    This class is fundamental to:
    - Unit movement planning
    - Resource gathering logistics
    - City placement strategy
    - Spatial relationships in observation encoding

    Args:
        x (int): X-coordinate
        y (int): Y-coordinate

    Note:
        Uses Manhattan distance for all calculations since
        units can only move horizontally or vertically.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, pos) -> int:
        """Computes Manhattan distance between positions.

        Args:
            pos (Position): Target position

        Returns:
            int: Manhattan distance (|dx| + |dy|)
        """
        return abs(pos.x - self.x) + abs(pos.y - self.y)

    def distance_to(self, pos):
        """Calculates Manhattan distance to another position.

        This is the primary distance metric used for:
        - Path planning
        - Resource targeting
        - City placement decisions

        Args:
            pos (Position): Target position

        Returns:
            int: Manhattan (L1/grid) distance
        """
        return self - pos

    def is_adjacent(self, pos):
        """Checks if another position is adjacent.

        Adjacent means exactly one cell away horizontally
        or vertically (not diagonally).

        Args:
            pos (Position): Position to check

        Returns:
            bool: True if positions are adjacent
        """
        return (self - pos) <= 1

    def __eq__(self, pos) -> bool:
        return self.x == pos.x and self.y == pos.y

    def equals(self, pos):
        return self == pos

    def translate(self, direction, units) -> 'Position':
        """Creates new position by moving in a direction.

        Used for:
        - Computing unit movements
        - Planning paths
        - Determining action targets

        Args:
            direction (DIRECTIONS): Direction to move
            units (int): Number of cells to move

        Returns:
            Position: New position after translation

        Note:
            DIRECTIONS.CENTER returns the same position,
            useful for 'stay still' actions.
        """
        if direction == DIRECTIONS.NORTH:
            return Position(self.x, self.y - units)
        elif direction == DIRECTIONS.EAST:
            return Position(self.x + units, self.y)
        elif direction == DIRECTIONS.SOUTH:
            return Position(self.x, self.y + units)
        elif direction == DIRECTIONS.WEST:
            return Position(self.x - units, self.y)
        elif direction == DIRECTIONS.CENTER:
            return Position(self.x, self.y)

    def direction_to(self, target_pos: 'Position') -> DIRECTIONS:
        """Determines optimal direction to reach target.

        Critical for:
        - Unit navigation
        - Resource collection pathing
        - City expansion planning
        - Combat positioning

        Args: 
            target_pos (Position): Destination position

        Returns:
            DIRECTIONS: Optimal direction (NORTH, SOUTH, EAST, WEST, CENTER)
            that minimizes Manhattan distance to target

        Note:
            Returns CENTER if no direction reduces distance,
            useful for determining if target is reached.
        """
        check_dirs = [
            DIRECTIONS.NORTH,
            DIRECTIONS.EAST,
            DIRECTIONS.SOUTH,
            DIRECTIONS.WEST,
        ]
        closest_dist = self.distance_to(target_pos)
        closest_dir = DIRECTIONS.CENTER
        for direction in check_dirs:
            newpos = self.translate(direction, 1)
            dist = target_pos.distance_to(newpos)
            if dist < closest_dist:
                closest_dir = direction
                closest_dist = dist
        return closest_dir

    def astuple(self):
        return self.x, self.y

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"{self.__class__.__name__}: {str(self)}"
