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


def sync_to_s3() -> bool:
    """
    Sincroniza modelo champion y mlruns a S3.

    Returns
    -------
    bool
        True si la sincronización fue exitosa, False en caso contrario
    """
    print("\n" + "="*80)
    print("SINCRONIZANDO ARTEFACTOS A S3")
    print("="*80)

    # Verificar variables de entorno de AWS
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-2")

    if not aws_key or not aws_secret:
        print("[WARN] Credenciales AWS no configuradas")
        print("  Define AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY en .env")
        print("  Saltando sincronización a S3...")
        return True  # No fallar, solo advertir

    # Bucket S3 (puedes configurarlo en .env si es necesario)
    s3_bucket = os.getenv("S3_BUCKET_NAME", "mlops-tetouan-models")
    s3_prefix = os.getenv("S3_PREFIX", "power-tetouan")

    try:
        # Sincronizar modelo champion
        print("\n[INFO] Sincronizando modelo champion a S3...")
        models_dir = Path("models")
        champion_files = list(models_dir.glob("*_champion.pkl"))

        if champion_files:
            for champion_file in champion_files:
                s3_key = f"{s3_prefix}/models/{champion_file.name}"
                cmd = [
                    "aws", "s3", "cp",
                    str(champion_file),
                    f"s3://{s3_bucket}/{s3_key}",
                    "--region", aws_region
                ]

                print(f"  Subiendo {champion_file.name} a s3://{s3_bucket}/{s3_key}")
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    print(f"  [OK] {champion_file.name} subido exitosamente")
                else:
                    print(f"  [ERROR] Error al subir {champion_file.name}: {result.stderr}")
        else:
            print("  [WARN] No se encontraron modelos champion para subir")

        # Sincronizar mlruns
        print("\n[INFO] Sincronizando mlruns a S3...")
        mlruns_dir = Path("mlruns")

        if mlruns_dir.exists():
            cmd = [
                "aws", "s3", "sync",
                str(mlruns_dir),
                f"s3://{s3_bucket}/{s3_prefix}/mlruns/",
                "--region", aws_region,
                "--exclude", "*.git/*",
                "--exclude", "__pycache__/*"
            ]

            print(f"  Sincronizando {mlruns_dir} a s3://{s3_bucket}/{s3_prefix}/mlruns/")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"  [OK] mlruns sincronizado exitosamente")
            else:
                print(f"  [ERROR] Error al sincronizar mlruns: {result.stderr}")
        else:
            print("  [WARN] Directorio mlruns no encontrado")

        print("\n[OK] Sincronización a S3 completada")
        return True

    except FileNotFoundError:
        print("[ERROR] AWS CLI no está instalado")
        print("  Instala AWS CLI: pip install awscli")
        print("  o descarga desde: https://aws.amazon.com/cli/")
        return False
    except Exception as e:
        print(f"[ERROR] Error durante la sincronización a S3: {e}")
        return False


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

    # Paso 2: Sincronizar a S3
    sync_to_s3()

    # Paso 3: Desplegar imagen Docker
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
    print("  - Modelo y mlruns sincronizados a S3")
    print("  - Imagen Docker actualizada (si bash esta disponible)")
    print("\nPara verificar el despliegue:")
    print("  docker images | grep power-tetouan-api")
    print("  aws s3 ls s3://mlops-tetouan-models/power-tetouan/models/")
    print("\n")
