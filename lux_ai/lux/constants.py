class Constants:
    """
    Core game constants defining various aspects of the Lux AI game mechanics.
    
    This class contains nested classes that define:
    - Input/output message constants for game communication
    - Movement directions for units
    - Unit types (workers and carts)
    - Resource types and their collection order
    """

    class INPUT_CONSTANTS:
        """
        Constants used for parsing game state input messages.
        
        These string constants are used as keys in the game's
        input/output protocol for communicating game state between
        the agent and the game engine.
        """
        RESEARCH_POINTS = "rp"  # Research points accumulated
        RESOURCES = "r"         # Resources on the map
        UNITS = "u"            # Unit positions and states
        CITY = "c"             # City states and fuel levels
        CITY_TILES = "ct"      # Individual city tile data
        ROADS = "ccd"          # Road levels on cells
        DONE = "D_DONE"        # End of turn marker

    class DIRECTIONS:
        """
        Constants defining possible movement directions for units.
        
        Units can move in four cardinal directions (NORTH, SOUTH, EAST, WEST)
        or stay in place (CENTER). These are used in movement commands and
        pathfinding logic.
        """
        NORTH = "n"   # Move up (-y direction)
        WEST = "w"    # Move left (-x direction)
        SOUTH = "s"   # Move down (+y direction)
        EAST = "e"    # Move right (+x direction)
        CENTER = "c"  # Stay in current position

        @staticmethod
        def astuple(include_center: bool) -> tuple:
            """
            Get movement directions as a tuple.
            
            Args:
                include_center (bool): Whether to include the CENTER direction
                
            Returns:
                tuple: Tuple of direction constants, optionally including CENTER
                
            Note:
                Direction order is NORTH, EAST, SOUTH, WEST, [CENTER]
                This order is important for consistent movement patterns.
            """
            move_directions = (
                Constants.DIRECTIONS.NORTH,
                Constants.DIRECTIONS.EAST,
                Constants.DIRECTIONS.SOUTH,
                Constants.DIRECTIONS.WEST
            )
            if include_center:
                return move_directions + (Constants.DIRECTIONS.CENTER,)
            else:
                return move_directions

    class UNIT_TYPES:
        """
        Constants defining the types of units in the game.
        
        There are two unit types:
        - Workers (0): Can build cities and carry resources
        - Carts (1): Can carry more resources but cannot build
        
        These values are used to identify unit types in the game state
        and when creating new units.
        """
        WORKER = 0  # Worker unit type (can build cities)
        CART = 1    # Cart unit type (larger cargo capacity)

    class RESOURCE_TYPES:
        """
        Constants defining the types of resources in the game.
        
        Resources have different properties:
        - Wood: Basic resource, always available
        - Coal: Requires research to collect
        - Uranium: Requires more research to collect, highest fuel value
        
        The order in astuple() represents the typical research progression.
        """
        WOOD = "wood"        # Basic resource, no research needed
        URANIUM = "uranium"  # Advanced resource, requires most research
        COAL = "coal"       # Intermediate resource

        @staticmethod
        def astuple() -> tuple:
            """
            Get resource types as a tuple in collection order.
            
            Returns:
                tuple: (WOOD, COAL, URANIUM) in order of typical collection progression
                
            Note: 
                Order represents typical research progression:
                1. Wood (available immediately)
                2. Coal (requires 50 research points)
                3. Uranium (requires 200 research points)
            """
            return Constants.RESOURCE_TYPES.WOOD, Constants.RESOURCE_TYPES.COAL, Constants.RESOURCE_TYPES.URANIUM
