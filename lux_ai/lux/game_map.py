from typing import List, Optional

from .constants import Constants

# Game movement and resource constants
DIRECTIONS = Constants.DIRECTIONS  # North, South, East, West, Center
RESOURCE_TYPES = Constants.RESOURCE_TYPES  # Wood, Coal, Uranium


class Resource:
    """
    Represents a resource tile on the game map.
    
    Contains information about the type of resource (wood/coal/uranium)
    and the remaining amount that can be harvested.
    """
    def __init__(self, r_type: str, amount: int):
        """
        Initialize a resource tile.
        
        Args:
            r_type (str): Type of resource (wood/coal/uranium)
            amount (int): Amount of resource remaining
        """
        self.type = r_type
        self.amount = amount


class Cell:
    """
    Represents a single cell on the game map.
    
    Each cell can contain:
    - A resource (wood, coal, or uranium)
    - A city tile
    - A road (with varying levels of development)
    """
    def __init__(self, x, y):
        """
        Initialize a map cell.
        
        Args:
            x (int): X coordinate on the map
            y (int): Y coordinate on the map
        """
        self.pos = Position(x, y)
        self.resource: Optional[Resource] = None  # Resource on this cell, if any
        self.citytile = None  # City tile on this cell, if any
        self.road = 0  # Road level (0 = no road, higher values = faster movement)

    def has_resource(self) -> bool:
        """
        Check if this cell contains a harvestable resource.
        
        Returns:
            bool: True if the cell contains a resource with amount > 0
        """
        return self.resource is not None and self.resource.amount > 0


class GameMap:
    """
    Represents the game map grid and provides methods to access and modify cells.
    
    The map is a 2D grid of Cell objects, each potentially containing resources,
    city tiles, and roads. The coordinate system has (0,0) in the top-left corner
    with x increasing eastward and y increasing southward.
    """
    def __init__(self, width: int, height: int):
        """
        Initialize a new game map with the given dimensions.
        
        Args:
            width (int): Width of the map (number of columns)
            height (int): Height of the map (number of rows)
        """
        self.height = height
        self.width = width
        # Initialize 2D grid of cells
        self.map: List[List[Cell]] = [None] * height
        for y in range(0, self.height):
            self.map[y] = [None] * width
            for x in range(0, self.width):
                self.map[y][x] = Cell(x, y)

    def get_cell_by_pos(self, pos) -> Cell:
        """
        Get the cell at a given Position object.
        
        Args:
            pos (Position): Position object containing x,y coordinates
        
        Returns:
            Cell: The cell at the specified position
        """
        return self.map[pos.y][pos.x]

    def get_cell(self, x: int, y: int) -> Cell:
        """
        Get the cell at the specified x,y coordinates.
        
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
        
        Returns:
            Cell: The cell at the specified coordinates
        """
        return self.map[y][x]

    def _setResource(self, r_type: str, x: int, y: int, amount: int):
        """
        Internal method to set or update a resource in a cell.
        
        Args:
            r_type (str): Type of resource (wood/coal/uranium)
            x (int): X coordinate
            y (int): Y coordinate
            amount (int): Amount of resource to set
            
        Note:
            This is an internal method used by the game engine to track state.
            It should not be called directly by agents or external code.
        """
        cell = self.get_cell(x, y)
        cell.resource = Resource(r_type, amount)


class Position:
    """
    Represents a position on the game map and provides methods for position manipulation.
    
    This class handles:
    - Position comparison and equality checking
    - Distance calculations (Manhattan/L1 distance)
    - Movement and direction calculations
    - Position translation in cardinal directions
    """
    def __init__(self, x: int, y: int):
        """
        Initialize a position with x,y coordinates.
        
        Args:
            x (int): X coordinate (increases eastward)
            y (int): Y coordinate (increases southward)
        """
        self.x = x
        self.y = y

    def __sub__(self, pos) -> int:
        """
        Calculate Manhattan distance between two positions.
        
        Args:
            pos (Position): Position to calculate distance to
            
        Returns:
            int: Manhattan (L1/grid) distance between positions
        """
        return abs(pos.x - self.x) + abs(pos.y - self.y)

    def distance_to(self, pos) -> int:
        """
        Calculate Manhattan (L1/grid) distance to another position.
        
        Args:
            pos (Position): Target position
            
        Returns:
            int: Manhattan distance to target position
        """
        return self - pos

    def is_adjacent(self, pos) -> bool:
        """
        Check if this position is adjacent to another position.
        
        Args:
            pos (Position): Position to check adjacency with
            
        Returns:
            bool: True if positions are adjacent (including diagonally)
        """
        return (self - pos) <= 1

    def __eq__(self, pos) -> bool:
        """
        Check if two positions are at the same coordinates.
        
        Args:
            pos (Position): Position to compare with
            
        Returns:
            bool: True if positions have same x,y coordinates
        """
        return self.x == pos.x and self.y == pos.y

    def equals(self, pos) -> bool:
        """
        Alternative method to check position equality.
        
        Args:
            pos (Position): Position to compare with
            
        Returns:
            bool: True if positions have same x,y coordinates
        """
        return self == pos

    def translate(self, direction, units) -> 'Position':
        """
        Create a new position by moving in a specified direction.
        
        Args:
            direction (DIRECTIONS): Direction to move (NORTH/SOUTH/EAST/WEST/CENTER)
            units (int): Number of cells to move
            
        Returns:
            Position: New position after translation
        """
        if direction == DIRECTIONS.NORTH: 
            return Position(self.x, self.y - units)  # Move up (decrease y)
        elif direction == DIRECTIONS.EAST:
            return Position(self.x + units, self.y)  # Move right (increase x)
        elif direction == DIRECTIONS.SOUTH:
            return Position(self.x, self.y + units)  # Move down (increase y)
        elif direction == DIRECTIONS.WEST:
            return Position(self.x - units, self.y)  # Move left (decrease x)
        elif direction == DIRECTIONS.CENTER:
            return Position(self.x, self.y)  # Stay in place

    def direction_to(self, target_pos: 'Position') -> DIRECTIONS:
        """
        Determine the optimal direction to move toward a target position.
        
        Args:
            target_pos (Position): Position to move toward
            
        Returns:
            DIRECTIONS: Best direction to move to get closer to target_pos
            
        Note:
            This method checks all four cardinal directions and returns the
            direction that most reduces the distance to the target position.
            If no direction reduces distance, returns CENTER.
        """
        # Check all cardinal directions
        check_dirs = [
            DIRECTIONS.NORTH,
            DIRECTIONS.EAST,
            DIRECTIONS.SOUTH,
            DIRECTIONS.WEST,
        ]
        closest_dist = self.distance_to(target_pos)
        closest_dir = DIRECTIONS.CENTER
        
        # Find direction that most reduces distance to target
        for direction in check_dirs:
            newpos = self.translate(direction, 1)
            dist = target_pos.distance_to(newpos)
            if dist < closest_dist:
                closest_dir = direction
                closest_dist = dist
        return closest_dir

    def astuple(self) -> tuple:
        """
        Convert position to a tuple of coordinates.
        
        Returns:
            tuple: (x, y) coordinate pair
        """
        return self.x, self.y

    def __str__(self) -> str:
        """
        Get string representation of position.
        
        Returns:
            str: Position as string in format "(x, y)"
        """
        return f"({self.x}, {self.y})"

    def __repr__(self) -> str:
        """
        Get detailed string representation of position.
        
        Returns:
            str: Detailed position string including class name
        """
        return f"{self.__class__.__name__}: {str(self)}"
