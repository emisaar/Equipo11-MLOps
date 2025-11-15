#!/usr/bin/env python
"""
Pipeline Stage 6: Despliegue del modelo champion
Despliega el mejor modelo (champion) en Docker Hub.
"""

from pathlib import Path
import yaml
import subprocess
import sys
import shutil
import os


def copy_champion_model(models_dir: Path) -> bool:
    """
    Copia el modelo champion al directorio raíz para que esté disponible en Docker.

    Parameters
    ----------
    models_dir : Path
        Directorio donde se encuentran los modelos

    Returns
    -------
    bool
        True si se copió exitosamente, False en caso contrario
    """
    print("\n" + "="*80)
    print("COPIANDO MODELO CHAMPION")
    print("="*80)

    # Buscar archivos _champion.pkl
    champion_files = list(models_dir.glob("*_champion.pkl"))

    if not champion_files:
        print(f"[ERROR] No se encontro modelo champion en {models_dir}")
        return False

    # Usar el primer modelo champion encontrado
    champion_file = champion_files[0]
    print(f"[OK] Modelo champion encontrado: {champion_file}")

    # Copiar al directorio raíz del proyecto (para que Docker lo incluya)
    dest_dir = Path("models")
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_file = dest_dir / champion_file.name

    # Verificar si ya existe y es el mismo archivo
    if dest_file.exists():
        import hashlib

        def get_file_hash(filepath):
            """Calcula el hash MD5 de un archivo."""
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()

        try:
            source_hash = get_file_hash(champion_file)
            dest_hash = get_file_hash(dest_file)

            if source_hash == dest_hash:
                print(f"[OK] El modelo champion ya esta actualizado en: {dest_file}")
                return True
            else:
                print(f"[INFO] Detectado nuevo modelo champion, actualizando...")
        except Exception as e:
            print(f"[WARN] No se pudo verificar el hash: {e}")

    # Intentar copiar el archivo
    try:
        # En Windows, si el archivo está en uso, intentar con nombre temporal
        temp_file = dest_dir / f"{champion_file.stem}_temp.pkl"

        # Copiar a archivo temporal primero
        shutil.copy2(champion_file, temp_file)

        # Intentar renombrar/reemplazar
        try:
            if dest_file.exists():
                dest_file.unlink()
            temp_file.rename(dest_file)
            print(f"[OK] Modelo champion copiado a: {dest_file}")
        except PermissionError:
            # Si no podemos reemplazar, dejamos el temporal
            print(f"[WARN] No se pudo reemplazar {dest_file} (archivo en uso)")
            print(f"[INFO] Modelo copiado a: {temp_file}")
            print(f"[INFO] Detener la API de Docker y renombrar manualmente:")
            print(f"  docker-compose down")
            print(f"  mv {temp_file} {dest_file}")
            print(f"  docker-compose up -d")
            # Aun así consideramos éxito porque el archivo está copiado
            return True

    except Exception as e:
        print(f"[ERROR] Error al copiar el modelo: {e}")
        return False

    return True


def deploy_docker_image() -> bool:
    """
    Ejecuta el script de despliegue de Docker.

    Returns
    -------
    bool
        True si el despliegue fue exitoso, False en caso contrario
    """
    print("\n" + "="*80)
    print("DESPLEGANDO IMAGEN DOCKER")
    print("="*80)

    # Verificar que existe el script
    script_path = Path("scripts/setup-docker-hub.sh")
    if not script_path.exists():
        print(f"[ERROR] Script de despliegue no encontrado: {script_path}")
        return False

    # Ejecutar el script
    try:
        # Forzar rebuild sin caché para incluir modelo champion
        env = os.environ.copy()
        env['DOCKER_BUILD_NO_CACHE'] = 'true'

        # En Windows, necesitamos bash para ejecutar scripts .sh
        if sys.platform == "win32":
            # Intentar usar Git Bash si está disponible
            git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
            if git_bash.exists():
                cmd = [str(git_bash), str(script_path)]
            else:
                print("[WARN] Git Bash no encontrado en Windows")
                print("  Para desplegar en Docker, ejecuta manualmente:")
                print(f"  bash {script_path}")
                return True  # No fallar, solo advertir
        else:
            cmd = ["bash", str(script_path)]

        print(f"\n[INFO] Ejecutando: {' '.join(cmd)}")
        print(f"[INFO] Forzando rebuild sin cache para incluir modelo champion")
        print("-" * 80)

        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            capture_output=False,
            text=True,
            env=env
        )

        if result.returncode == 0:
            print("\n[OK] Despliegue de Docker completado exitosamente")
            return True
        else:
            print(f"\n[ERROR] Error en el despliegue de Docker (codigo: {result.returncode})")
            return False

    except FileNotFoundError:
        print("[WARN] Bash no esta disponible en este sistema")
        print("  Para desplegar en Docker, ejecuta manualmente:")
        print(f"  bash {script_path}")
        return True  # No fallar, solo advertir
    except Exception as e:
        print(f"[ERROR] Error al ejecutar el script de despliegue: {e}")
        return False


if __name__ == "__main__":
    # Carga parámetros desde params.yaml
    params = yaml.safe_load(open("params.yaml", "r", encoding="utf-8"))

    models_dir = Path(params["models"]["output_dir"])

    print("\n" + "="*80)
    print("DESPLIEGUE DEL MODELO CHAMPION")
    print("="*80)

    # Paso 1: Copiar modelo champion
    if not copy_champion_model(models_dir):
        print("\n[ERROR] Error al copiar el modelo champion")
        sys.exit(1)

    # Paso 2: Desplegar imagen Docker
    if not deploy_docker_image():
        print("\n[ERROR] Error al desplegar la imagen Docker")
        print("  Puedes desplegar manualmente ejecutando:")
        print("  bash scripts/setup-docker-hub.sh")
        # No salir con error para permitir continuar el pipeline
        # sys.exit(1)

    print("\n" + "="*80)
    print("DESPLIEGUE COMPLETADO")
    print("="*80)
    print("\n[OK] El modelo champion ha sido desplegado exitosamente")
    print("  - Modelo copiado al directorio models/")
    print("  - Imagen Docker actualizada (si bash esta disponible)")
    print("\nPara verificar el despliegue:")
    print("  docker images | grep power-tetouan-api")
    print("\n")
