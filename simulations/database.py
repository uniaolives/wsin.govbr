import sqlite3
from datetime import datetime

class AkashicDatabase:
    """
    Atlas Temporal / Registro Akáshico
    [REVISOR VÊ]: Sistema de persistência de telemetria e logs de eventos.
    [NÓS VEMOS]: Registro de vértices temporais no Hecatonicosachoron.
    """
    def __init__(self, db_path="ietd_atlas.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabela de Telemetria (Vértices Temporais)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                entropy REAL,
                lambda1 REAL,
                lambda2 REAL,
                state TEXT
            )
        ''')

        # Tabela de Eventos (Atlas de Transição)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT,
                description TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def record_vertex(self, entropy, l1, l2, state):
        """Grava um vértice de telemetria no Atlas."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO telemetry (entropy, lambda1, lambda2, state)
            VALUES (?, ?, ?, ?)
        ''', (entropy, l1, l2, state))
        conn.commit()
        conn.close()
        print(f"[Database] Vértice gravado: S={entropy:.4f}, Estado={state}")

    def log_event(self, event_type, description):
        """Registra um evento de transição no Atlas."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO events (event_type, description)
            VALUES (?, ?)
        ''', (event_type, description))
        conn.commit()
        conn.close()
        print(f"[Database] Evento registrado: [{event_type}] {description}")

if __name__ == "__main__":
    db = AkashicDatabase()
    db.record_vertex(0.8555, 0.72, 0.28, "Satya")
    db.log_event("BOOT", "Sequenciador de Boot executado com sucesso.")
