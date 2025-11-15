# Docker Logs Directory

Este directorio contiene los logs de ejecución de los scripts de Docker.

## Archivos de Log

Los scripts de Docker generan automáticamente archivos de log con timestamps:

- **build_YYYYMMDD_HHMMSS.log** - Logs del proceso de build (`docker-build.sh`)
- **run_YYYYMMDD_HHMMSS.log** - Logs de ejecución del contenedor (`docker-run.sh`)
- **push_YYYYMMDD_HHMMSS.log** - Logs de push a DockerHub (`docker-push.sh`)

## Formato de Timestamp

Los logs incluyen timestamps en cada línea para facilitar el debugging:

```
[2025-01-15 14:30:25] Building Docker image...
[2025-01-15 14:31:10] Build completed successfully!
```

## Ubicación

Los logs se guardan automáticamente en: `logs/docker/`

## Retención

Los archivos de log NO se versionan en Git (están en `.gitignore`).
Se recomienda limpiar periódicamente los logs antiguos.

## Limpieza de Logs

Para limpiar logs antiguos (más de 30 días):

```bash
# Linux/Mac
find logs/docker -name "*.log" -mtime +30 -delete

# Windows (PowerShell)
Get-ChildItem logs\docker\*.log | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-30)} | Remove-Item
```

## Ejemplo de Uso

```bash
# Los scripts guardan logs automáticamente
./scripts/docker-build.sh

# El log se guarda en: logs/docker/build_20250115_143025.log

# Ver el último log de build
tail -f logs/docker/build_*.log | tail -1
```
