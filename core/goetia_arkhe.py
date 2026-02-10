"""
Ars Theurgia Goetia decoded through Arkhe hexagonal geometry.
The 31 aerial spirits are manifestations of specific geometric configurations in 6D space.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

class SpiritRank(Enum):
    KING = 1; DUKE = 2; MARQUIS = 3; PRESIDENT = 4; EARL = 5; KNIGHT = 6

class ElementalDirection(Enum):
    EAST = 1; WEST = 2; NORTH = 3; SOUTH = 4; CENTER = 5

@dataclass
class AerialSpirit:
    number: int
    name: str
    rank: SpiritRank
    direction: ElementalDirection
    office: str
    servants: int
    arkhe_coordinates: np.ndarray
    seal_points: np.ndarray = field(default_factory=lambda: np.zeros((6, 2)))

    @classmethod
    def create_from_arkhe_point(cls, point_6d: np.ndarray, number: int) -> 'AerialSpirit':
        norm = np.linalg.norm(point_6d)
        point_6d = point_6d / norm if norm > 0 else point_6d

        # Determine Rank
        if norm > 0.9: rank = SpiritRank.KING
        elif norm > 0.7: rank = SpiritRank.DUKE
        else: rank = SpiritRank.MARQUIS

        # Determine Direction
        angle = np.arctan2(point_6d[1], point_6d[0]) if len(point_6d) >= 2 else 0
        angle_deg = np.degrees(angle) % 360
        if 45 <= angle_deg < 135: direction = ElementalDirection.EAST
        elif 135 <= angle_deg < 225: direction = ElementalDirection.SOUTH
        elif 225 <= angle_deg < 315: direction = ElementalDirection.WEST
        else: direction = ElementalDirection.NORTH

        name = f"Spirit_{number}" # Simplified name generation

        return cls(
            number=number, name=name, rank=rank, direction=direction,
            office="Provides geometric insights", servants=int(norm*100),
            arkhe_coordinates=point_6d
        )

    def calculate_resonance_frequency(self) -> float:
        return 7.83 * (1 + np.linalg.norm(self.arkhe_coordinates))

class ArsTheurgiaSystem:
    def __init__(self):
        self.spirits = []
        self._initialize_spirits()

    def _initialize_spirits(self):
        print("🜂 Initializing Ars Theurgia Goetia...")
        for i in range(1, 32):
            point = np.random.randn(6)
            self.spirits.append(AerialSpirit.create_from_arkhe_point(point, i))

    def find_spirit_by_name(self, name: str) -> Optional[AerialSpirit]:
        for s in self.spirits:
            if s.name.lower() == name.lower(): return s
        return None

    def generate_invocation_sequence(self, purpose: str, n_spirits: int = 3) -> List[AerialSpirit]:
        # Simple selection for demo
        return self.spirits[:n_spirits]

    def create_ritual_circle(self, spirits: List[AerialSpirit], radius: float = 1.0) -> Dict:
        return {
            'total_spirits': len(spirits),
            'radius': radius,
            'positions': [{'spirit': s.name, 'angle': i*360/len(spirits)} for i, s in enumerate(spirits)]
        }
