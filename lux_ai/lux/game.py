from .constants import Constants
from .game_constants import GAME_CONSTANTS
from .game_map import GameMap
from .game_objects import Player, Unit, City

INPUT_CONSTANTS = Constants.INPUT_CONSTANTS
DN_CYCLE_LEN = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]


class Game:
    """Core game state manager for the Lux AI environment.

    This class maintains and updates the complete game state, including:
    - Map state (resources, roads, etc.)
    - Player states (units, cities, research)
    - Turn information and day/night cycle

    The Game class serves as the primary interface between the competition
    engine and the RL environment. It:
    1. Parses game state updates from the engine
    2. Maintains a consistent world state
    3. Provides access to game state components
    4. Tracks game progression (turns, day/night)

    Implementation Details:
    - Uses GameMap for spatial information
    - Manages Player objects for team states
    - Handles resource updates and city management
    - Processes unit actions and movements
    - Tracks research progress and city fuel

    Note:
        This implementation focuses on efficient state updates and
        clean interfaces for the RL environment to observe the
        game state.
    """

    def _initialize(self, messages):
        """Initializes the game state from initial messages.

        Sets up the game map, players, and initial state variables
        based on the first messages received from the game engine.

        Args:
            messages (list): Initial messages from game engine containing:
                - Player ID (int)
                - Map dimensions (str: "width height")
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
        """Signals the end of turn to the game engine.

        Outputs the "D_FINISH" command to indicate that all
        actions for the current turn have been submitted.

        Note:
            This is a critical synchronization point between
            the agent and the game engine.
        """
        print("D_FINISH")

    def _reset_player_states(self):
        """Resets all player-specific state variables.

        Clears unit lists, city dictionaries, and city tile counts for both players.
        Called at the start of each turn to prepare for state updates.

        Implementation:
        - Resets unit lists to empty
        - Clears city dictionaries
        - Zeros city tile counts
        - Maintains player IDs and research points
        """
        self.players[0].units = []
        self.players[0].cities = {}
        self.players[0].city_tile_count = 0
        self.players[1].units = []
        self.players[1].cities = {}
        self.players[1].city_tile_count = 0

    # noinspection PyProtectedMember
    def _update(self, messages):
        """Updates game state based on messages from the engine.

        Processes a series of state update messages to refresh the game state
        for the current turn. Handles updates for:
        - Research points
        - Resources on map
        - Unit positions and resources
        - City states and fuel
        - City tile positions and cooldowns
        - Road levels

        Args:
            messages (list): List of space-separated update strings, each containing:
                - Update type identifier
                - Type-specific data (positions, amounts, IDs, etc.)

        Implementation Details:
        1. Resets map and player states
        2. Processes updates in order:
           - Research points (team progress)
           - Resources (wood, coal, uranium)
           - Units (workers, carts)
           - Cities (fuel, upkeep)
           - City tiles (position, cooldown)
           - Roads (position, level)
        3. Maintains game turn counter
        4. Handles optional position skipping

        Note:
            The skip attribute allows certain positions to be ignored,
            which is useful for the RL environment to test hypothetical
            states.
        """
        self.map = GameMap(self.map_width, self.map_height)
        self.turn += 1
        self._reset_player_states()

        skip = getattr(self, "skip", None)

        for update in messages:
            if update == "D_DONE":
                break
            strs = update.split(" ")
            input_identifier = strs[0]
            if input_identifier == INPUT_CONSTANTS.RESEARCH_POINTS:
                team = int(strs[1])
                self.players[team].research_points = int(strs[2])
            elif input_identifier == INPUT_CONSTANTS.RESOURCES:
                r_type = strs[1]
                x = int(strs[2])
                y = int(strs[3])
                amt = int(float(strs[4]))
                if (x, y) != skip:
                    self.map._setResource(r_type, x, y, amt)
            elif input_identifier == INPUT_CONSTANTS.UNITS:
                unittype = int(strs[1])
                team = int(strs[2])
                unitid = strs[3]
                x = int(strs[4])
                y = int(strs[5])
                cooldown = float(strs[6])
                wood = int(strs[7])
                coal = int(strs[8])
                uranium = int(strs[9])
                if (x, y) != skip:
                    self.players[team].units.append(Unit(team, unittype, unitid, x, y, cooldown, wood, coal, uranium))
            elif input_identifier == INPUT_CONSTANTS.CITY:
                team = int(strs[1])
                cityid = strs[2]
                fuel = float(strs[3])
                lightupkeep = float(strs[4])
                self.players[team].cities[cityid] = City(team, cityid, fuel, lightupkeep)
            elif input_identifier == INPUT_CONSTANTS.CITY_TILES:
                team = int(strs[1])
                cityid = strs[2]
                x = int(strs[3])
                y = int(strs[4])
                cooldown = float(strs[5])
                city = self.players[team].cities[cityid]
                if (x, y) != skip:
                    citytile = city._add_city_tile(x, y, cooldown)
                    self.map.get_cell(x, y).citytile = citytile
                    self.players[team].city_tile_count += 1
            elif input_identifier == INPUT_CONSTANTS.ROADS:
                x = int(strs[1])
                y = int(strs[2])
                road = float(strs[3])
                if (x, y) != skip:
                    self.map.get_cell(x, y).road = road

    @property
    def is_night(self) -> bool:
        """Determines if the current turn is during night time.

        Uses the day/night cycle length and current turn to calculate
        whether it's currently night. This affects:
        - City fuel consumption
        - Unit cooldowns
        - Resource collection rates

        Returns:
            bool: True if it's night time, False if day time

        Note:
            The day/night cycle is a key strategic element as it affects
            resource gathering efficiency and city survival.
        """
        return self.turn % DN_CYCLE_LEN >= GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]
