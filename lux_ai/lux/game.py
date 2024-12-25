from .constants import Constants
from .game_constants import GAME_CONSTANTS
from .game_map import GameMap
from .game_objects import Player, Unit, City

# Constants for processing game input and managing day/night cycle
INPUT_CONSTANTS = Constants.INPUT_CONSTANTS
# Total length of a day/night cycle in game turns
DN_CYCLE_LEN = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]


class Game:
    """
    Main game state manager for the Lux AI competition.
    
    This class handles initialization of the game state, processes turn updates,
    and maintains the current state of the game including:
    - Map state (resources, roads, etc.)
    - Player states (units, cities, research points)
    - Turn counter and day/night cycle
    
    The game state is updated each turn based on input messages from the game engine,
    which provide information about resources, units, cities, and other game elements.
    """
    def _initialize(self, messages):
        """
        Initialize the game state from the starting game messages.
        
        Args:
            messages (list[str]): List of initialization messages from the game engine.
                messages[0]: Player ID (0 or 1)
                messages[1]: Map dimensions in format "width height"
        
        Sets up:
        - Player ID
        - Map dimensions and GameMap instance
        - Initial Player objects for both teams
        """
        self.id = int(messages[0])
        self.turn = -1
        # get some other necessary initial input
        map_info = messages[1].split(" ")
        self.map_width = int(map_info[0])
        self.map_height = int(map_info[1])
        self.map = GameMap(self.map_width, self.map_height)
        self.players = [Player(0), Player(1)]

    @staticmethod
    def _end_turn():
        """
        Signal to the game engine that this turn's commands are complete.
        
        Prints the "D_FINISH" command to indicate the agent has finished its turn.
        """
        print("D_FINISH")

    def _reset_player_states(self):
        """
        Reset both players' unit lists, city dictionaries, and city tile counts.
        
        Called at the start of each turn before processing updates to ensure
        the game state is fresh and prevent duplicate units/cities.
        """
        # Reset player 0's state
        self.players[0].units = []
        self.players[0].cities = {}
        self.players[0].city_tile_count = 0
        # Reset player 1's state
        self.players[1].units = []
        self.players[1].cities = {}
        self.players[1].city_tile_count = 0

    # noinspection PyProtectedMember
    def _update(self, messages):
        """
        Process turn update messages to refresh the game state.
        
        Args:
            messages (list[str]): List of update messages from the game engine describing
                                the current state of resources, units, cities, etc.
        
        Updates:
        - Game map with new resource amounts
        - Players' research points
        - Unit positions, stats, and cargo
        - City locations, fuel, and light upkeep
        - Road levels on the map
        
        The update process continues until a "D_DONE" message is received, indicating
        all state updates for the current turn have been processed.
        """
        self.map = GameMap(self.map_width, self.map_height)
        self.turn += 1
        self._reset_player_states()

        skip = getattr(self, "skip", None)

        # Process each update message until "D_DONE" is encountered
        for update in messages:
            if update == "D_DONE":
                break
            strs = update.split(" ")
            input_identifier = strs[0]

            # Update research points for a team
            if input_identifier == INPUT_CONSTANTS.RESEARCH_POINTS:
                team = int(strs[1])
                self.players[team].research_points = int(strs[2])

            # Update resource (wood/coal/uranium) amounts on the map
            elif input_identifier == INPUT_CONSTANTS.RESOURCES:
                r_type = strs[1]  # Resource type (wood, coal, uranium)
                x = int(strs[2])
                y = int(strs[3])
                amt = int(float(strs[4]))  # Amount of resource
                if (x, y) != skip:
                    self.map._setResource(r_type, x, y, amt)

            # Create or update unit positions and cargo
            elif input_identifier == INPUT_CONSTANTS.UNITS:
                unittype = int(strs[1])  # Type of unit (worker/cart)
                team = int(strs[2])      # Team ID (0/1)
                unitid = strs[3]         # Unique unit identifier
                x = int(strs[4])
                y = int(strs[5])
                cooldown = float(strs[6]) # Turns until unit can act again
                # Unit's cargo of each resource type
                wood = int(strs[7])
                coal = int(strs[8])
                uranium = int(strs[9])
                if (x, y) != skip:
                    self.players[team].units.append(Unit(team, unittype, unitid, x, y, cooldown, wood, coal, uranium))

            # Create or update city stats (fuel and upkeep)
            elif input_identifier == INPUT_CONSTANTS.CITY:
                team = int(strs[1])
                cityid = strs[2]         # Unique city identifier
                fuel = float(strs[3])    # Current fuel amount
                lightupkeep = float(strs[4])  # Fuel consumed per turn
                self.players[team].cities[cityid] = City(team, cityid, fuel, lightupkeep)

            # Update individual city tiles within cities
            elif input_identifier == INPUT_CONSTANTS.CITY_TILES: 
                team = int(strs[1])
                cityid = strs[2]         # ID of the city this tile belongs to
                x = int(strs[3])
                y = int(strs[4])
                cooldown = float(strs[5]) # Turns until tile can act again
                city = self.players[team].cities[cityid]
                if (x, y) != skip:
                    # Add the city tile to both the city and the map
                    citytile = city._add_city_tile(x, y, cooldown)
                    self.map.get_cell(x, y).citytile = citytile
                    self.players[team].city_tile_count += 1

            # Update road levels on the map
            elif input_identifier == INPUT_CONSTANTS.ROADS:
                x = int(strs[1])
                y = int(strs[2])
                road = float(strs[3])    # Road level (affects movement speed)
                if (x, y) != skip:
                    self.map.get_cell(x, y).road = road

    @property
    def is_night(self) -> bool:
        """
        Check if the current turn is during the night phase.
        
        Returns:
            bool: True if it's night time, False if it's day time.
            
        The day/night cycle is determined by the turn number modulo the total cycle length.
        If the remainder is greater than or equal to the day length, it's night time.
        """
        return self.turn % DN_CYCLE_LEN >= GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]
