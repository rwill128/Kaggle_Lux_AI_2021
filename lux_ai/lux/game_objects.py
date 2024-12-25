from typing import Dict, List, Optional

from .constants import Constants
from .game_map import Position
from .game_constants import GAME_CONSTANTS

# Unit type constants (WORKER, CART)
UNIT_TYPES = Constants.UNIT_TYPES


class Player:
    """
    Represents a player in the game and manages their resources, units, and cities.
    
    Each player has:
    - Research points for unlocking resource types
    - A list of units (workers and carts)
    - A dictionary of cities
    - A count of total city tiles owned
    """
    def __init__(self, team):
        """
        Initialize a player for the given team.
        
        Args:
            team (int): Team identifier (0 or 1)
        """
        self.team = team
        self.research_points = 0  # Points from city tile research actions
        self.units: List[Unit] = []  # List of units owned by this player
        self.cities: Dict[str, City] = {}  # Cities indexed by city ID
        self.city_tile_count = 0  # Total number of city tiles owned

    def researched_coal(self) -> bool:
        """
        Check if the player has researched coal harvesting capability.
        
        Returns:
            bool: True if player has enough research points to mine coal
        """
        return self.research_points >= GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["COAL"]

    def researched_uranium(self) -> bool:
        """
        Check if the player has researched uranium harvesting capability.
        
        Returns:
            bool: True if player has enough research points to mine uranium
        """
        return self.research_points >= GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"]

    @property
    def city_tiles(self) -> List['CityTile']:
        """
        Get a list of all city tiles owned by this player across all cities.
        
        Returns:
            List[CityTile]: List of all city tiles owned by the player
        """
        return [ct for city in self.cities.values() for ct in city.citytiles]

    def get_unit_by_id(self, unit_id: str) -> Optional['Unit']:
        """
        Find a unit by its unique identifier.
        
        Args:
            unit_id (str): Unique identifier of the unit to find
            
        Returns:
            Optional[Unit]: The unit with matching ID, or None if not found
        """
        for unit in self.units:
            if unit_id == unit.id:
                return unit
        return None


class City:
    """
    Represents a city in the game, which consists of connected city tiles.
    
    Cities require fuel to survive and have an upkeep cost per turn.
    They can be expanded by building additional city tiles.
    """
    def __init__(self, teamid: int, cityid: str, fuel: float, light_upkeep: float):
        """
        Initialize a city.
        
        Args:
            teamid (int): Team that owns this city (0 or 1)
            cityid (str): Unique identifier for this city
            fuel (float): Current fuel stored in the city
            light_upkeep (float): Fuel consumed per turn to maintain the city
        """
        self.cityid = cityid
        self.team = teamid
        self.fuel = fuel  # Current fuel reserves
        self.citytiles: List[CityTile] = []  # List of tiles that make up this city
        self.light_upkeep = light_upkeep  # Fuel consumed per turn

    def _add_city_tile(self, x: int, y: int, cooldown: float) -> 'CityTile':
        """
        Internal method to add a new city tile to this city.
        
        Args:
            x (int): X coordinate of the new tile
            y (int): Y coordinate of the new tile
            cooldown (float): Initial cooldown of the tile
            
        Returns:
            CityTile: The newly created city tile
            
        Note:
            This is an internal method used by the game engine.
            It should not be called directly by agents.
        """
        ct = CityTile(self.team, self.cityid, x, y, cooldown)
        self.citytiles.append(ct)
        return ct

    def get_light_upkeep(self) -> float:
        """
        Get the amount of fuel consumed per turn by this city.
        
        Returns:
            float: Fuel consumption rate per turn
        """
        return self.light_upkeep

    def __str__(self) -> str:
        """
        Get string representation of the city (its ID).
        
        Returns:
            str: City ID
        """
        return self.cityid

    def __repr__(self) -> str:
        """
        Get detailed string representation of the city.
        
        Returns:
            str: Detailed city string including class name
        """
        return f"City: {str(self)}"


class CityTile:
    """
    Represents a single tile of a city. City tiles can:
    - Conduct research to unlock new resource types
    - Build new worker units
    - Build new cart units
    
    Each action has a cooldown period before the tile can act again.
    """
    def __init__(self, teamid: int, cityid: str, x: int, y: int, cooldown: float):
        """
        Initialize a city tile.
        
        Args:
            teamid (int): Team that owns this tile (0 or 1)
            cityid (str): ID of the city this tile belongs to
            x (int): X coordinate on the map
            y (int): Y coordinate on the map
            cooldown (float): Current cooldown before tile can act again
        """
        self.cityid = cityid
        self.team = teamid
        self.pos = Position(x, y)  # Position on the map
        self.cooldown = cooldown  # Turns until next action allowed

    def can_act(self) -> bool:
        """
        Check if this tile can perform an action this turn.
        
        Actions include:
        - Research
        - Building workers
        - Building carts
        
        Returns:
            bool: True if cooldown is less than 1, indicating an action is possible
        """
        return self.cooldown < 1

    def research(self) -> str:
        """
        Generate command string to research technology this turn.
        
        Research points contribute to unlocking new resource types:
        - Coal requires 50 points
        - Uranium requires 200 points
        
        Returns:
            str: Command string in format "r x y"
        """
        return "r {} {}".format(self.pos.x, self.pos.y)

    def build_worker(self) -> str:
        """
        Generate command string to build a worker unit this turn.
        
        Workers can:
        - Move around the map
        - Collect resources
        - Build new city tiles
        
        Returns:
            str: Command string in format "bw x y"
        """
        return "bw {} {}".format(self.pos.x, self.pos.y)

    def build_cart(self) -> str:
        """
        Generate command string to build a cart unit this turn.
        
        Carts can:
        - Move around the map
        - Carry more resources than workers
        - Transfer resources to other units
        
        Returns:
            str: Command string in format "bc x y"
        """
        return "bc {} {}".format(self.pos.x, self.pos.y)

    def __str__(self) -> str:
        """
        Get string representation of the city tile.
        
        Returns:
            str: City tile string with position
        """
        return f"CityTile: {self.pos}"

    def __repr__(self) -> str:
        """
        Get detailed string representation of the city tile.
        
        Returns:
            str: Same as __str__ for city tiles
        """
        return str(self)

    def __hash__(self) -> int:
        """
        Generate hash value for the city tile.
        
        The hash combines the city ID and position to uniquely identify the tile.
        
        Returns:
            int: Hash value
        """
        return hash(f"{self.cityid}_{self.pos}")

    def __eq__(self, other) -> bool:
        """
        Check if two city tiles are equal.
        
        Args:
            other: Another object to compare with
            
        Returns:
            bool: True if other is a CityTile with same position and city ID
        """
        return isinstance(other, CityTile) and self.pos == other.pos and self.cityid == other.cityid


class Cargo:
    """
    Represents the resource cargo carried by a unit (worker or cart).
    
    Tracks quantities of three resource types:
    - Wood (basic resource, always available)
    - Coal (requires research to collect)
    - Uranium (requires research to collect)
    """
    def __init__(self):
        """
        Initialize an empty cargo hold.
        
        All resource quantities start at 0.
        """
        self.wood = 0     # Amount of wood carried
        self.coal = 0     # Amount of coal carried
        self.uranium = 0  # Amount of uranium carried

    def get(self, resource_type: str) -> int:
        """
        Get the amount of a specific resource type in cargo.
        
        Args:
            resource_type (str): Type of resource to check (wood/coal/uranium)
            
        Returns:
            int: Amount of the specified resource
            
        Raises:
            ValueError: If resource_type is not recognized
        """
        if resource_type == Constants.RESOURCE_TYPES.WOOD:
            return self.wood
        elif resource_type == Constants.RESOURCE_TYPES.COAL:
            return self.coal
        elif resource_type == Constants.RESOURCE_TYPES.URANIUM:
            return self.uranium
        else:
            raise ValueError(f"Unrecognized resource_type: {resource_type}")

    def __str__(self) -> str:
        """
        Get string representation of cargo contents.
        
        Returns:
            str: Cargo contents showing amounts of each resource
        """
        return f"Cargo | Wood: {self.wood}, Coal: {self.coal}, Uranium: {self.uranium}"


class Unit:
    """
    Represents a mobile unit in the game (worker or cart).
    
    Units can:
    - Move around the map
    - Carry resources (wood, coal, uranium)
    - Build cities (workers only)
    - Transfer resources between units
    - Pillage roads
    
    Each unit has a cooldown period between actions and a maximum cargo capacity
    that varies between workers and carts.
    """
    def __init__(self, teamid: int, u_type: str, unitid: str, x: int, y: int, 
                 cooldown: float, wood: int, coal: int, uranium: int):
        """
        Initialize a unit.
        
        Args:
            teamid (int): Team that owns this unit (0 or 1)
            u_type (str): Type of unit (WORKER or CART)
            unitid (str): Unique identifier for this unit
            x (int): Initial X coordinate
            y (int): Initial Y coordinate
            cooldown (float): Initial cooldown value
            wood (int): Initial wood cargo
            coal (int): Initial coal cargo
            uranium (int): Initial uranium cargo
        """
        self.pos = Position(x, y)  # Current position
        self.team = teamid
        self.id = unitid
        self.type = u_type
        self.cooldown = cooldown  # Turns until next action allowed
        # Initialize cargo with starting resources
        self.cargo = Cargo()
        self.cargo.wood = wood
        self.cargo.coal = coal
        self.cargo.uranium = uranium

    def is_worker(self) -> bool:
        """
        Check if this unit is a worker.
        
        Workers can:
        - Carry fewer resources than carts
        - Build new city tiles
        
        Returns:
            bool: True if unit is a worker
        """
        return self.type == UNIT_TYPES.WORKER

    def is_cart(self) -> bool:
        """
        Check if this unit is a cart.
        
        Carts can:
        - Carry more resources than workers
        - Cannot build cities
        
        Returns:
            bool: True if unit is a cart
        """
        return self.type == UNIT_TYPES.CART

    def get_cargo_space_left(self) -> int:
        """
        Calculate remaining cargo capacity for this unit.
        
        Capacity varies by unit type:
        - Workers: 100 units
        - Carts: 2000 units
        
        Returns:
            int: Remaining cargo space
        """
        spaceused = self.cargo.wood + self.cargo.coal + self.cargo.uranium
        if self.type == UNIT_TYPES.WORKER:
            return GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] - spaceused
        else:
            return GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"] - spaceused

    def can_build(self, game_map) -> bool:
        """
        Check if this unit can build a city tile at its current position.
        
        Requirements:
        - Must be a worker
        - Cell must not contain resources
        - Unit must be able to act (cooldown < 1)
        - Must have enough resources (wood/coal/uranium) to meet city cost
        
        Args:
            game_map: Current game map state
            
        Returns:
            bool: True if city can be built
        """
        cell = game_map.get_cell_by_pos(self.pos)
        if not cell.has_resource() and self.can_act() and (self.cargo.wood + self.cargo.coal + self.cargo.uranium) >= \
                GAME_CONSTANTS["PARAMETERS"]["CITY_BUILD_COST"]:
            return True
        return False

    def can_act(self) -> bool:
        """
        Check if this unit can perform an action this turn.
        
        Actions include:
        - Moving
        - Building
        - Transferring resources
        - Pillaging
        
        Note: This does not check for collisions with other units or cities.
        
        Returns:
            bool: True if cooldown is less than 1
        """
        return self.cooldown < 1

    def move(self, dir) -> str:
        """
        Generate command string to move in a direction.
        
        Args: 
            dir (DIRECTIONS): Direction to move (NORTH/SOUTH/EAST/WEST)
            
        Returns:
            str: Command string in format "m unitid direction"
        """
        return "m {} {}".format(self.id, dir)

    def transfer(self, dest_id: str, resourceType: str, amount: int) -> str:
        """
        Generate command string to transfer resources to another unit.
        
        Args:
            dest_id (str): ID of the destination unit
            resourceType (str): Type of resource to transfer (wood/coal/uranium)
            amount (int): Amount of resource to transfer
            
        Returns:
            str: Command string in format "t source_id dest_id resource_type amount"
        """
        return "t {} {} {} {}".format(self.id, dest_id, resourceType, amount)

    def build_city(self) -> str:
        """
        Generate command string to build a new city tile.
        
        The city will be built at the unit's current position.
        Only workers can build cities.
        
        Returns:
            str: Command string in format "bcity unitid"
        """
        return "bcity {}".format(self.id)

    def pillage(self) -> str:
        """
        Generate command string to pillage (destroy) a road.
        
        Pillaging reduces the road level at the unit's current position,
        making movement slower for all units.
        
        Returns:
            str: Command string in format "p unitid"
        """
        return "p {}".format(self.id)

    def __str__(self) -> str:
        """
        Get string representation of the unit.
        
        Returns:
            str: Unit details including team, type, and position
        """
        return f"Unit: team_{self.team}/type_{self.type}/pos_{self.pos}"

    def __repr__(self) -> str:
        """
        Get detailed string representation of the unit.
        
        Returns:
            str: Same as __str__ for units
        """
        return str(self)

    def __hash__(self) -> int:
        """
        Generate hash value for the unit.
        
        Uses the unit's unique ID for hashing.
        
        Returns:
            int: Hash value
        """
        return hash(self.id)

    def __eq__(self, other) -> bool:
        """
        Check if two units are equal.
        
        Args:
            other: Another object to compare with
            
        Returns:
            bool: True if other is a Unit with the same ID
        """
        return isinstance(other, Unit) and self.id == other.id
