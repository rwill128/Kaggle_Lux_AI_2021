from typing import Dict, List, Optional

from .constants import Constants
from .game_map import Position
from .game_constants import GAME_CONSTANTS

UNIT_TYPES = Constants.UNIT_TYPES


class Player:
    """Represents a player in the game and their current state.

    This class maintains all state information for a player, including:
    - Research progress (affects resource collection abilities)
    - Units (workers and carts)
    - Cities and city tiles
    
    The Player class is central to the RL observation space as it
    encodes the current state and capabilities of each agent.

    Args:
        team (int): Player's team ID (0 or 1)

    Attributes:
        team (int): Player's team ID
        research_points (int): Current research progress
        units (List[Unit]): List of player's units
        cities (Dict[str, City]): Map of city ID to City objects
        city_tile_count (int): Total number of city tiles owned
    """
    def __init__(self, team):
        self.team = team
        self.research_points = 0
        self.units: List[Unit] = []
        self.cities: Dict[str, City] = {}
        self.city_tile_count = 0

    def researched_coal(self) -> bool:
        """Checks if player can collect coal resources.

        Coal collection requires reaching a research threshold.
        This affects both observation space (what resources are
        collectible) and action space (what tiles are harvestable).

        Returns:
            bool: True if coal can be collected
        """
        return self.research_points >= GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["COAL"]

    def researched_uranium(self) -> bool:
        """Checks if player can collect uranium resources.

        Uranium collection requires reaching a higher research threshold
        than coal. Uranium provides more fuel value but requires more
        research investment.

        Returns:
            bool: True if uranium can be collected
        """
        return self.research_points >= GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"]

    @property
    def city_tiles(self) -> List['CityTile']:
        return [ct for city in self.cities.values() for ct in city.citytiles]

    def get_unit_by_id(self, unit_id: str) -> Optional['Unit']:
        """Retrieves a unit by its unique identifier.

        Used in the RL environment for:
        - Action validation
        - Resource transfer targeting
        - Unit tracking across turns

        Args:
            unit_id (str): Unique unit identifier

        Returns:
            Optional[Unit]: Unit if found, None otherwise
        """
        for unit in self.units:
            if unit_id == unit.id:
                return unit
        return None


class City:
    """Represents a city in the game world.

    Cities are collections of city tiles that:
    - Consume fuel to survive
    - Enable unit production
    - Provide research points
    - Generate game-winning light

    Cities are critical strategic elements in the RL environment:
    - Their survival directly impacts reward
    - They enable expansion through unit production
    - They represent long-term investment (fuel management)

    Args:
        teamid (int): Owning team's ID
        cityid (str): Unique city identifier
        fuel (float): Current fuel amount
        light_upkeep (float): Fuel consumed per turn

    Attributes:
        cityid (str): Unique city identifier
        team (int): Owning team's ID
        fuel (float): Current fuel amount
        citytiles (List[CityTile]): Component tiles
        light_upkeep (float): Fuel consumed per turn
    """
    def __init__(self, teamid, cityid, fuel, light_upkeep):
        self.cityid = cityid
        self.team = teamid
        self.fuel = fuel
        self.citytiles: List[CityTile] = []
        self.light_upkeep = light_upkeep

    def _add_city_tile(self, x, y, cooldown):
        ct = CityTile(self.team, self.cityid, x, y, cooldown)
        self.citytiles.append(ct)
        return ct

    def get_light_upkeep(self):
        return self.light_upkeep

    def __str__(self):
        return self.cityid

    def __repr__(self):
        return f"City: {str(self)}"


class CityTile: 
    """Represents a single tile of a city.

    CityTiles are the fundamental building blocks of cities that:
    - Can build new units (workers/carts)
    - Can conduct research
    - Must be supplied with fuel to survive
    - Contribute to victory through light generation

    In the RL environment, CityTiles are both:
    - Part of the observation space (position, cooldown)
    - Sources of actions (build, research)

    Args:
        teamid (int): Owning team's ID
        cityid (str): Parent city's ID
        x (int): X-coordinate
        y (int): Y-coordinate
        cooldown (float): Turns until next action

    Attributes:
        cityid (str): Parent city's ID
        team (int): Owning team's ID
        pos (Position): Tile's position
        cooldown (float): Turns until next action
    """
    def __init__(self, teamid, cityid, x, y, cooldown):
        self.cityid = cityid
        self.team = teamid
        self.pos = Position(x, y)
        self.cooldown = cooldown

    def can_act(self) -> bool:
        """
        Whether or not this unit can research or build
        """
        return self.cooldown < 1

    def research(self) -> str:
        """
        returns command to ask this tile to research this turn
        """
        return "r {} {}".format(self.pos.x, self.pos.y)

    def build_worker(self) -> str:
        """
        returns command to ask this tile to build a worker this turn
        """
        return "bw {} {}".format(self.pos.x, self.pos.y)

    def build_cart(self) -> str:
        """
        returns command to ask this tile to build a cart this turn
        """
        return "bc {} {}".format(self.pos.x, self.pos.y)

    def __str__(self):
        return f"CityTile: {self.pos}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(f"{self.cityid}_{self.pos}")

    def __eq__(self, other):
        return isinstance(other, CityTile) and self.pos == other.pos and self.cityid == other.cityid


class Cargo:
    """Represents resources carried by a unit.

    Manages the resource inventory for units, tracking:
    - Wood (basic fuel)
    - Coal (advanced fuel, requires research)
    - Uranium (premium fuel, requires advanced research)

    This class is important for the RL environment as:
    - Resource amounts affect valid actions
    - Cargo capacity influences gathering strategy
    - Resource types have different fuel values

    Attributes:
        wood (int): Amount of wood carried
        coal (int): Amount of coal carried
        uranium (int): Amount of uranium carried
    """
    def __init__(self):
        self.wood = 0
        self.coal = 0
        self.uranium = 0

    def get(self, resource_type: str) -> int:
        if resource_type == Constants.RESOURCE_TYPES.WOOD:
            return self.wood
        elif resource_type == Constants.RESOURCE_TYPES.COAL:
            return self.coal
        elif resource_type == Constants.RESOURCE_TYPES.URANIUM:
            return self.uranium
        else:
            raise ValueError(f"Unrecognized resource_type: {resource_type}")

    def __str__(self) -> str:
        return f"Cargo | Wood: {self.wood}, Coal: {self.coal}, Uranium: {self.uranium}"


class Unit:
    """Represents a mobile unit in the game.

    Units are the primary actors in the game that can:
    - Move around the map
    - Gather resources
    - Build cities
    - Transfer resources
    - Pillage roads

    Units are central to the RL action space:
    - Movement in 4 directions
    - Resource collection
    - City building
    - Resource transfer
    - Road destruction

    The unit state is also key in the observation space:
    - Position
    - Cargo contents
    - Cooldown status
    - Unit type (worker/cart)

    Args:
        teamid (int): Owning team's ID
        u_type (UNIT_TYPES): Unit type (WORKER/CART)
        unitid (str): Unique unit identifier
        x (int): X-coordinate
        y (int): Y-coordinate
        cooldown (float): Turns until next action
        wood (int): Initial wood cargo
        coal (int): Initial coal cargo
        uranium (int): Initial uranium cargo

    Attributes:
        pos (Position): Unit's position
        team (int): Owning team's ID
        id (str): Unique identifier
        type (UNIT_TYPES): Unit type
        cooldown (float): Action cooldown
        cargo (Cargo): Resource inventory
    """
    def __init__(self, teamid, u_type, unitid, x, y, cooldown, wood, coal, uranium):
        self.pos = Position(x, y)
        self.team = teamid
        self.id = unitid
        self.type = u_type
        self.cooldown = cooldown
        self.cargo = Cargo()
        self.cargo.wood = wood
        self.cargo.coal = coal
        self.cargo.uranium = uranium

    def is_worker(self) -> bool:
        return self.type == UNIT_TYPES.WORKER

    def is_cart(self) -> bool:
        return self.type == UNIT_TYPES.CART

    def get_cargo_space_left(self):
        """Calculates remaining cargo capacity.

        Critical for RL action selection:
        - Determines if resource collection is possible
        - Influences resource gathering strategy
        - Affects city building capability

        Returns:
            int: Remaining cargo space (different for workers vs carts)
        """
        spaceused = self.cargo.wood + self.cargo.coal + self.cargo.uranium
        if self.type == UNIT_TYPES.WORKER:
            return GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] - spaceused
        else:
            return GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"] - spaceused

    def can_build(self, game_map) -> bool:
        """Checks if unit can build a city at current position.

        Key part of action space validation:
        - Requires sufficient resources
        - Location must be empty
        - Unit must be ready to act

        Args:
            game_map (GameMap): Current game map state

        Returns:
            bool: True if city can be built at current position
        """
        cell = game_map.get_cell_by_pos(self.pos)
        if not cell.has_resource() and self.can_act() and (self.cargo.wood + self.cargo.coal + self.cargo.uranium) >= \
                GAME_CONSTANTS["PARAMETERS"]["CITY_BUILD_COST"]:
            return True
        return False

    def can_act(self) -> bool:
        """Checks if unit is ready to perform an action.

        Critical for action space determination:
        - Affects movement availability
        - Controls resource gathering timing
        - Determines city building opportunity
        - Influences combat timing

        Note:
            Does not check for collisions or other game rules,
            only cooldown status.

        Returns:
            bool: True if cooldown < 1, indicating unit can act
        """
        return self.cooldown < 1

    def move(self, dir) -> str:
        """Generates movement command string.

        Core part of action space:
        - Primary unit navigation command
        - Used for resource gathering
        - Used for strategic positioning
        - Critical for city expansion

        Args:
            dir (DIRECTIONS): Direction to move (NORTH/SOUTH/EAST/WEST)

        Returns:
            str: Formatted movement command for game engine
        """
        return "m {} {}".format(self.id, dir)

    def transfer(self, dest_id, resourceType, amount) -> str:
        """Generates resource transfer command string.

        Strategic action for resource management:
        - Enables resource sharing between units
        - Critical for city fueling strategy
        - Allows specialized resource gathering roles
        - Facilitates efficient city building

        Args:
            dest_id (str): Target unit's ID
            resourceType (str): Resource type (WOOD/COAL/URANIUM)
            amount (int): Amount to transfer

        Returns:
            str: Formatted transfer command for game engine
        """
        return "t {} {} {} {}".format(self.id, dest_id, resourceType, amount)

    def build_city(self) -> str:
        """Generates city building command string.

        Strategic expansion action that:
        - Creates new unit production points
        - Establishes resource processing centers
        - Generates victory points through light
        - Requires significant resource investment

        Returns:
            str: Formatted city build command for game engine
        """
        return "bcity {}".format(self.id)

    def pillage(self) -> str:
        """Generates road pillaging command string.

        Tactical action that:
        - Reduces enemy movement speed
        - Disrupts resource supply lines
        - Controls map territory
        - Creates defensive barriers

        Returns:
            str: Formatted pillage command for game engine
        """
        return "p {}".format(self.id)

    def __str__(self):
        return f"Unit: team_{self.team}/type_{self.type}/pos_{self.pos}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Unit) and self.id == other.id
