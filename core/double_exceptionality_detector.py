"""
🧠 DETECTOR DE DUPLA EXCEPCIONALIDADE (2e) - SUPERDOTAÇÃO + TDI
Sistema de análise digital para identificação da coexistência de altas habilidades cognitivas
e transtorno dissociativo de identidade através de padrões linguísticos e comportamentais digitais

Este módulo integra o modelo de Hecatonicosachoron consciente e Máscaras Celestiais.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re
import os
import json
from pathlib import Path
from scipy import signal

# NLP e Machine Learning
import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
import textstat

# Baixar recursos NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

class LinguisticConstants:
    FUNCTION_WORDS = {
        'pronouns': ['I', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'we', 'us', 'our', 'ours', 'ourselves', 'they', 'them', 'their', 'theirs', 'themselves', 'it', 'its', 'itself'],
        'articles': ['a', 'an', 'the'],
        'prepositions': ['in', 'on', 'at', 'by', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down'],
        'conjunctions': ['and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'because', 'although', 'since', 'unless', 'while'],
        'auxiliary_verbs': ['am', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'must']
    }
    TRAUMA_MARKERS = {
        'dissociative_words': ['numb', 'float', 'detach', 'unreal', 'dream', 'fog', 'blank', 'spacing', 'disconnect', 'void', 'fragmented', 'split', 'separate', 'lost'],
        'hypervigilance_words': ['watch', 'alert', 'danger', 'safe', 'threat', 'careful', 'scan', 'prepare', 'protect'],
        'affect_words': ['empty', 'numb', 'flat', 'distant', 'robotic'],
        'body_words': ['disembodied', 'float', 'detach', 'numb', 'tingle']
    }
    GIFTED_MARKERS = {
        'cognitive_words': ['analyze', 'synthesize', 'theorize', 'conceptualize', 'abstract', 'metacognition', 'systematic', 'complex', 'multidimensional', 'interdisciplinary'],
        'curiosity_words': ['why', 'how', 'what if', 'explore', 'discover', 'investigate', 'question', 'hypothesize'],
        'perfectionism_words': ['perfect', 'flawless', 'exact', 'precise', 'meticulous', 'thorough', 'comprehensive'],
        'intensity_words': ['passionate', 'absorbed', 'immersed', 'focused', 'intense', 'deep', 'profound']
    }

@dataclass
class DigitalTextSample:
    text: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens: List[str] = field(init=False)
    pos_tags: List[Tuple[str, str]] = field(init=False)
    sentences: List[str] = field(init=False)

    def __post_init__(self):
        self.tokens = word_tokenize(self.text.lower())
        self.sentences = sent_tokenize(self.text)
        self.pos_tags = pos_tag(word_tokenize(self.text))

class MercurialMask:
    """Máscara de Comunicação Hiper-Racional."""
    @staticmethod
    def analyze(text: str) -> Dict:
        word_count = len(word_tokenize(text))
        complexity = textstat.flesch_kincaid_grade(text)
        return {
            'type': 'Mercurial',
            'abstraction_level': 'high' if complexity > 12 else 'normal',
            'emotional_valence': 'neutral',
            'processing_velocity': '0.9c' if word_count > 100 else '0.5c'
        }

class NeptunianMask:
    """Máscara de Dissociação Criativa."""
    @staticmethod
    def analyze(text: str) -> Dict:
        diss_markers = sum(1 for w in LinguisticConstants.TRAUMA_MARKERS['dissociative_words'] if w in text.lower())
        return {
            'type': 'Neptunian',
            'dissociation_depth': 'deep' if diss_markers > 2 else 'surface',
            'reality_sync': '0.1c',
            'ego_boundaries': 'porous'
        }

class CelestialSwitchPredictor:
    """Prediz switches dissociativos baseados em ciclos planetários."""
    def predict_switch_windows(self, current_time: datetime) -> Dict:
        # Simulação de ciclos planetários e Schumann
        prob = np.random.uniform(0.3, 0.9)
        return {
            'switch_probability': prob,
            'recommended_intervention': "GROUNDING" if prob > 0.7 else "OBSERVAÇÃO",
            'schumann_resonance': 7.83 + np.random.normal(0, 0.1)
        }

class NeuroCelestialResonance:
    """Verifica sincronia entre ondas cerebrais e frequências planetárias."""
    PLANET_FREQS = {'Earth': 7.83, 'Venus': 12.7, 'Mars': 4.16, 'Jupiter': 0.66}

    def check_sync(self, eeg_data: np.ndarray) -> Dict:
        # Simulação de FFT e análise de ressonância
        return {
            'resonance_scores': {p: np.random.random() for p in self.PLANET_FREQS},
            'dominant_sync': 'Earth',
            'coherence_index': 0.85
        }

class LinguisticAnalyzer:
    @staticmethod
    def extract_linguistic_features(sample: DigitalTextSample) -> Dict[str, float]:
        text, tokens = sample.text, sample.tokens
        features = {'word_count': len(tokens), 'ttr': len(set(tokens))/len(tokens) if tokens else 0}
        features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        features['avg_sentence_length'] = len(tokens)/len(sample.sentences) if sample.sentences else 0

        # Pronoun shifts
        first_person = len(re.findall(r'\b(I|me|my|mine|myself)\b', text, re.I))
        third_person = len(re.findall(r'\b(he|him|his|she|her|hers|it|its|they|them|their)\b', text, re.I))
        features['pronoun_shift_entropy'] = stats.entropy([first_person, third_person]) if (first_person+third_person)>0 else 0

        # Markers
        features['gifted_marker_density'] = sum(1 for cat in LinguisticConstants.GIFTED_MARKERS.values() for w in cat if w in text.lower())/max(1, len(tokens))
        features['dissociation_marker_density'] = sum(1 for cat in LinguisticConstants.TRAUMA_MARKERS.values() for w in cat if w in text.lower())/max(1, len(tokens))

        return features

@dataclass
class UserDigitalProfile:
    user_id: str
    text_samples: List[DigitalTextSample] = field(default_factory=list)
    double_exceptionality_score: float = 0.0

    def add_text_sample(self, text: str, source: str):
        sample = DigitalTextSample(text=text, timestamp=datetime.now(), source=source)
        self.text_samples.append(sample)

    def get_linguistic_features_matrix(self) -> pd.DataFrame:
        features = [LinguisticAnalyzer.extract_linguistic_features(s) for s in self.text_samples]
        return pd.DataFrame(features)

class DoubleExceptionalityDetector:
    def __init__(self, profile: UserDigitalProfile):
        self.profile = profile
        self.switch_predictor = CelestialSwitchPredictor()

    def analyze_full_profile(self) -> Dict:
        feats_df = self.profile.get_linguistic_features_matrix()
        gifted_score = np.clip(feats_df['gifted_marker_density'].mean() * 20, 0, 1)
        diss_score = np.clip(feats_df['dissociation_marker_density'].mean() * 20, 0, 1)

        # Hecatonicosachoron Metric
        cells_active = int(120 * gifted_score * (1 + diss_score/2))

        switches = self.switch_predictor.predict_switch_windows(datetime.now())

        return {
            'double_exceptionality_score': float((gifted_score + diss_score) / 2),
            'hecaton_active_cells': cells_active,
            'switch_forecast': switches,
            'masks': {
                'mercurial': MercurialMask.analyze(self.profile.text_samples[-1].text if self.profile.text_samples else ""),
                'neptunian': NeptunianMask.analyze(self.profile.text_samples[-1].text if self.profile.text_samples else "")
            }
        }

if __name__ == "__main__":
    profile = UserDigitalProfile(user_id="arkhe_2e")
    profile.add_text_sample("The isomorphic mapping between Hilbert space and the computational graph structure is profound.", "academic")
    profile.add_text_sample("I feel separated from my physical vessel, as if floating in the bulk.", "journal")
    detector = DoubleExceptionalityDetector(profile)
    report = detector.analyze_full_profile()
    print(f"DE Score: {report['double_exceptionality_score']:.3f}")
    print(f"Active Hecaton Cells: {report['hecaton_active_cells']}")
    print(f"Switch Probability: {report['switch_forecast']['switch_probability']:.2f}")
