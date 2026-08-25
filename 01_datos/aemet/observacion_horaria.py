"""
Consulta datos de observación horaria (tiempo casi real) de una estación
meteorológica de AEMET OpenData.

Uso:
    1. Define tu API Key como variable de entorno antes de ejecutar:
         export AEMET_API_KEY="tu_api_key_aqui"      (Linux/Mac/WSL)
         $env:AEMET_API_KEY="tu_api_key_aqui"         (PowerShell)
    2. Ejecuta:
         python observacion_horaria.py CLE

NOTA DE SEGURIDAD:
    No escribas la API Key directamente en este archivo si vas a compartirlo
    o subirlo a un repositorio. Usa siempre una variable de entorno o un
    archivo .env excluido de git.
"""

import os
import sys
import json
import requests

BASE_URL = "https://opendata.aemet.es/opendata/api"


def get_api_key() -> str:
    api_key = os.environ.get("AEMET_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: no se encontró la variable de entorno AEMET_API_KEY.\n"
            "Defínela antes de ejecutar el script, por ejemplo:\n"
            '  export AEMET_API_KEY="tu_api_key"'
        )
    return api_key


def consultar_estacion(idema: str, api_key: str) -> list[dict]:
    """
    Devuelve la lista de observaciones horarias de la estación indicada
    (últimas ~24h, que es lo que ofrece este endpoint).
    """
    url = f"{BASE_URL}/observacion/convencional/datos/estacion/{idema}"
    headers = {"api_key": api_key}

    # Paso 1: la API devuelve metadatos con una URL temporal a los datos reales
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    meta = resp.json()

    if meta.get("estado") != 200:
        raise RuntimeError(f"AEMET devolvió un error: {meta}")

    datos_url = meta["datos"]

    # Paso 2: descargamos los datos reales desde la URL temporal
    datos_resp = requests.get(datos_url, timeout=15)
    datos_resp.raise_for_status()
    # AEMET a veces devuelve el JSON con encoding latin-1
    datos_resp.encoding = "latin-1"
    return datos_resp.json()


def main():
    idema = sys.argv[1] if len(sys.argv) > 1 else "3195"  # 3195 = Madrid Retiro, ejemplo
    api_key = get_api_key()

    print(f"Consultando estación '{idema}'...")
    datos = consultar_estacion(idema, api_key)

    if not datos:
        print("No se han recibido datos para esa estación.")
        return

    print(f"\nSe han recibido {len(datos)} registros.\n")
    for registro in datos[-5:]:  # mostramos los 5 más recientes
        fecha = registro.get("fint", "¿?")
        temp = registro.get("ta", "¿?")
        hum = registro.get("hr", "¿?")
        viento = registro.get("vv", "¿?")
        precip = registro.get("prec", "¿?")
        print(
            f"{fecha} | Temp: {temp}°C | Humedad: {hum}% | "
            f"Viento: {viento} m/s | Precip: {precip} mm"
        )

    # Guardamos el JSON completo por si lo quieres procesar luego
    out_path = f"observacion_{idema}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
    print(f"\nDatos completos guardados en: {out_path}")


if __name__ == "__main__":
    main()
