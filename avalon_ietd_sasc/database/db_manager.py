from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, Telemetry, Event

class DatabaseManager:
    """
    [REVISOR VÊ]: Gerencia a persistência via SQLAlchemy.
    [NÓS VEMOS]: Guardião do Registro Akáshico.
    """
    def __init__(self, db_url="sqlite:///ietd_system.db"):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def initialize(self):
        Base.metadata.create_all(self.engine)
        print("   Banco de dados inicializado via SQLAlchemy.")

    def add_telemetry(self, data):
        session = self.Session()
        new_entry = Telemetry(
            temperature=data.get('temperature'),
            ph=data.get('ph'),
            conductivity=data.get('conductivity'),
            x=data.get('x', 0.0),
            y=data.get('y', 0.0),
            z=data.get('z', 0.0),
            w=data.get('w', 0.0)
        )
        session.add(new_entry)
        session.commit()
        session.close()

    def add_event(self, event_type, description, severity=0):
        session = self.Session()
        new_event = Event(
            event_type=event_type,
            description=description,
            severity=severity
        )
        session.add(new_event)
        session.commit()
        session.close()

    def close(self):
        pass
