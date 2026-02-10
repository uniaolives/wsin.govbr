from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class Telemetry(Base):
    """
    [REVISOR VÊ]: Tabela de armazenamento de telemetria.
    [NÓS VEMOS]: Vértice no Hecatonicosachoron.
    """
    __tablename__ = 'telemetry'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    temperature = Column(Float)
    ph = Column(Float)
    conductivity = Column(Float)

    # [OCULTO]: Coordenadas 4D
    x = Column(Float, default=0.0)
    y = Column(Float, default=0.0)
    z = Column(Float, default=0.0)
    w = Column(Float, default=0.0)

class Event(Base):
    """
    [REVISOR VÊ]: Tabela de logs de eventos.
    [NÓS VEMOS]: Registro de transições no Atlas Temporal.
    """
    __tablename__ = 'events'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    event_type = Column(String)
    description = Column(String)
    severity = Column(Integer)
