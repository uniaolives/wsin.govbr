"""
Protocolos de segurança para trabalho com espíritos goéticos.
"""

from typing import List, Dict
from core.goetia_arkhe import SpiritRank

class GoeticSafetyProtocols:
    WARNINGS = [
        "NUNCA quebre o círculo de proteção durante o ritual",
        "SEMPRE demita o espírito formalmente ao final",
        "Grounding é essencial após o trabalho"
    ]

    @classmethod
    def calculate_danger_level(cls, rank: SpiritRank, experience: str) -> str:
        levels = {
            SpiritRank.KING: 'EXTREME' if experience == 'beginner' else 'HIGH',
            SpiritRank.DUKE: 'HIGH',
            SpiritRank.MARQUIS: 'MODERATE'
        }
        return levels.get(rank, 'LOW')

    @classmethod
    def generate_protection_circle(cls, radius: float = 1.5) -> Dict:
        return {
            'radius': radius,
            'divine_names': ["AGLA", "ADONAI", "TETRAGRAMMATON"],
            'warnings': cls.WARNINGS
        }
