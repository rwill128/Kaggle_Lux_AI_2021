"""
Visualization annotation utilities for the Lux AI game.

This module provides functions to create debug visualization commands that can be
used to draw shapes, markers, and text on the game map for debugging purposes.
Each function returns a string command that the game engine will interpret to
render the corresponding visualization element.
"""

def circle(x: int, y: int) -> str:
    """
    Create a command to draw a circle marker at the specified coordinates.
    
    Args:
        x (int): X-coordinate on the game map
        y (int): Y-coordinate on the game map
        
    Returns:
        str: Command string in format "dc x y" for drawing a circle
    """
    return f"dc {x} {y}"


def x(x: int, y: int) -> str:
    """
    Create a command to draw an X marker at the specified coordinates.
    
    Args:
        x (int): X-coordinate on the game map
        y (int): Y-coordinate on the game map
        
    Returns:
        str: Command string in format "dx x y" for drawing an X
    """
    return f"dx {x} {y}"


def line(x1: int, y1: int, x2: int, y2: int) -> str:
    """
    Create a command to draw a line between two points on the map.
    
    Useful for visualizing paths, connections, or ranges between game elements.
    
    Args:
        x1 (int): Starting X-coordinate
        y1 (int): Starting Y-coordinate
        x2 (int): Ending X-coordinate
        y2 (int): Ending Y-coordinate
        
    Returns:
        str: Command string in format "dl x1 y1 x2 y2" for drawing a line
    """
    return f"dl {x1} {y1} {x2} {y2}"


def text(x: int, y: int, message: str, fontsize: int = 16) -> str:
    """
    Create a command to display text at specific coordinates on the map.
    
    Spaces in the message are automatically replaced with underscores for
    compatibility with the visualization system.
    
    Args:
        x (int): X-coordinate on the game map
        y (int): Y-coordinate on the game map
        message (str): Text to display (spaces will be converted to underscores)
        fontsize (int, optional): Size of the text. Defaults to 16
        
    Returns:
        str: Command string in format "dt x y 'message' fontsize" for drawing text
    """
    message = message.replace(" ", "_")
    return f"dt {x} {y} '{message}' {fontsize}"


def sidetext(message: str) -> str:
    """
    Create a command to display text in the sidebar next to the map.
    
    Useful for showing game state information, debug messages, or other
    annotations that shouldn't clutter the main map view.
    
    Args:
        message (str): Text to display in the sidebar
        
    Returns:
        str: Command string in format "dst 'message'" for drawing sidebar text
    """
    return f"dst '{message}'"
