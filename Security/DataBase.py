from sqlalchemy.sql import func
from sqlalchemy.orm import Session # no delet
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column , Integer , String , DateTime , create_engine , UniqueConstraint

class ContentDataBase:
    def __call__(self, addres : str = "sqlite:///./Security/db/midelware.db") -> None:
        self.engine = create_engine( addres , connect_args={"check_same_thread": False} )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.Base = declarative_base()

database = ContentDataBase()
database()

class Model:
    class CsrfIpData(database.Base):
        __tablename__ = "CsrfTokenPro"
        id =          Column( Integer, primary_key=True, index=True )
        Ip =          Column( String )
        Headers =     Column( String )
        Hash =        Column( String )
        TimeCreated = Column(DateTime(timezone=True), server_default=func.now())
        TimeUpdated = Column(DateTime(timezone=True), onupdate=func.now())
        __table_args__ = (UniqueConstraint('Headers', name='_customer_location_uc'),)

    class BlockIp(database.Base):
        __tablename__ = "BlockApi"
        id =          Column(Integer, primary_key=True, index=True)
        Ip =          Column( String )
        TimeCreated = Column(DateTime(timezone=True), server_default=func.now())
        TimeUpdated = Column(DateTime(timezone=True), onupdate=func.now())

    
