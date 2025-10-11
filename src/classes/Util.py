import requests
import os
import json
import ast
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Añadimos SQLAlchemy para MySQL
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

class Util:
    # Motor de base de datos compartido
    _db_engine = None          
       
    @staticmethod
    def get_db_engine():
        """
        Inicializa y retorna un engine SQLAlchemy para MySQL local.
        Usa valores por defecto para conexión local si no existen en .env.
        """
        load_dotenv()
        if Util._db_engine:
            return Util._db_engine
        # Carga variables con valores por defecto local
        user = os.getenv("DB_USER")
        pwd  = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        name = os.getenv("DB_NAME")
        url = f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{name}"
        try:
            Util._db_engine = create_engine(url, pool_pre_ping=True)
        except SQLAlchemyError as e:
            raise EnvironmentError(f"No se pudo conectar a MySQL: {e}")
        return Util._db_engine
    
    @staticmethod
    def clean_name(c: str) -> str:
        """
        Normaliza nombres de columnas: minúsculas, espacios→guion_bajo.
        """
        return (c.strip().lower().replace("  ", " ").replace(" ", "_"))
    
class PowerConsumptionDAO:
    @staticmethod
    def fetch_data(page: int = 1, size: int = 10):
        """
        Recupera datos paginados de la tabla power_consumption.
        Retorna un dict con keys: items, total, page, size.
        """
        engine = Util.get_db_engine()
        offset = (page - 1) * size
        try:
            with engine.connect() as conn:
                total = conn.execute(text("SELECT COUNT(*) FROM power_consumption")).scalar_one()
                rows = conn.execute(
                    text("""
                        SELECT datetime, zone1, zone2, zone3, total_power, temperature, humidity
                        FROM power_consumption
                        ORDER BY datetime ASC
                        LIMIT :limit OFFSET :offset
                    """),
                    {"limit": size, "offset": offset}
                ).fetchall()
        except SQLAlchemyError as e:
            raise RuntimeError(f"Error al consultar la tabla power_consumption: {e}")
        
        items = [
            {
                "datetime": str(r[0]),
                "zone1": r[1],
                "zone2": r[2],
                "zone3": r[3],
                "total_power": r[4],
                "temperature": r[5],
                "humidity": r[6],
            } for r in rows
        ]
        
        return {"items": items, "total": total, "page": page, "size": size}