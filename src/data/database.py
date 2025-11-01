# database.py
# Gestión de conexiones a base de datos y acceso a datos mediante DAO
# ===========================

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


class Util:
    """
    Utilidades para gestión de base de datos.

    Provee métodos para inicializar y obtener engine de SQLAlchemy
    para conectarse a MySQL.
    """

    # Motor de base de datos compartido
    _db_engine = None

    @staticmethod
    def get_db_engine():
        """
        Inicializa y retorna un engine SQLAlchemy para MySQL local.
        Usa valores por defecto para conexión local si no existen en .env.

        Returns
        -------
        Engine
            Engine de SQLAlchemy configurado para MySQL

        Raises
        ------
        EnvironmentError
            Si no se puede conectar a MySQL
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


class PowerConsumptionDAO:
    """
    Data Access Object para la tabla power_consumption.

    Provee métodos para recuperar datos de consumo de energía
    desde la base de datos MySQL.
    """

    @staticmethod
    def fetch_data(page: int = 1, size: int = 10):
        """
        Recupera datos paginados de la tabla power_consumption.

        Parameters
        ----------
        page : int, default=1
            Número de página a recuperar
        size : int, default=10
            Cantidad de registros por página

        Returns
        -------
        dict
            Diccionario con keys: items, total, page, size
            - items: lista de registros como diccionarios
            - total: total de registros en la tabla
            - page: página actual
            - size: tamaño de página

        Raises
        ------
        RuntimeError
            Si hay un error al consultar la base de datos
        """
        engine = Util.get_db_engine()
        offset = (page - 1) * size

        try:
            with engine.connect() as conn:
                total = conn.execute(
                    text("SELECT COUNT(*) FROM power_consumption")
                ).scalar_one()

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
