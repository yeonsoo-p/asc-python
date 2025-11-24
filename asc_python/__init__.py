"""
ASC Python - ASCII Log File Reader with DBC Support

This package provides tools for reading and decoding ASC (ASCII log) files
using DBC (CAN Database) files. Data is stored as NumPy arrays for efficient analysis.

Main Classes:
    ASC: High-level interface for reading ASC files and accessing decoded CAN messages

Example:
    >>> from asc_python import ASC
    >>> # Single DBC for all channels (wildcard)
    >>> asc = ASC('recording.asc', [(-1, 'vehicle.dbc')])
    >>> # Access signal data
    >>> timestamps = asc.get_signal('GpsStatus', 'Time')
    >>> gps_mode = asc.get_signal('GpsStatus', 'GpsPosMode')
    >>> # Or use dictionary-style access
    >>> velocity = asc['Distance']['Distance']
    >>> # Get metadata
    >>> unit = asc.get_signal_unit('Distance', 'Distance')
"""

from .asc import ASC

__all__ = ["ASC"]
__version__ = "0.1.0"
