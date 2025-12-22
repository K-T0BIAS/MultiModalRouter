# exceptions.py
# Copyright (c) 2025 Tobias Karusseit
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

class FrozenException(Exception):
    def __init__(self, function: str) -> None:
        message: str = f"[RouteGraph] in '{function}' method: graph is frozen and can't be modified unless unfrozen first."
        super().__init__(message)