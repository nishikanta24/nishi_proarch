from sqlalchemy import Column, Integer, String, Float, JSON, Date, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Billing(Base):
    __tablename__ = 'billing'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_month = Column(String, nullable=False)
    account_id = Column(String, nullable=False)
    subscription = Column(String)
    service = Column(String, nullable=False)
    resource_group = Column(String)
    resource_id = Column(String)
    region = Column(String)
    usage_qty = Column(Float)
    unit_cost = Column(Float)
    cost = Column(Float, nullable=False)

class Resource(Base):
    __tablename__ = 'resources'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    resource_id = Column(String, unique=True, nullable=False)
    owner = Column(String)
    env = Column(String)
    tags_json = Column(JSON)

def get_engine(db_url):
    return create_engine(db_url)

def init_db(db_url):
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    return engine