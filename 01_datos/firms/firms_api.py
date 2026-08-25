#!/usr/bin/env python3
"""Descarga de detecciones FIRMS NRT con reintentos.

Un fallo de la API no puede confundirse con "no hay focos": los tres scripts
usaban un `except → DataFrame vacío` que dejaba `frp_max_50km_7d` y
`n_detec_50km_7d` a 0 para TODAS las estaciones, el XGB predecía con esas
features falseadas y el mapa se publicaba igual, sin ningún aviso (se ve en
los días sin fila en `rankings/verificacion_diaria.csv`: 2026-07-31, 08-02,
08-05). Aquí se reintenta con backoff y, si la API sigue sin responder CSV,
se levanta `FirmsCaido`: mejor que el workflow falle y quede el mapa de ayer
que publicar uno con las features de fuego a cero.

Un CSV con solo la cabecera SÍ es respuesta válida (0 detecciones reales) y
devuelve un DataFrame vacío sin error.
"""

import socket
import time
from io import StringIO

import pandas as pd
import requests
import urllib3.util.connection

# firms.modaps.eosdis.nasa.gov publica AAAA además de A, y los runners de
# GitHub Actions no tienen ruta IPv6: cuando getaddrinfo devuelve primero la
# AAAA, el connect muere al instante con `Network is unreachable` (ENETUNREACH
# — no es un timeout, de ahí que los 4 intentos se consuman en segundos). Pasó
# el 07-ago y el 15-ago. Forzar IPv4 evita esa ruta muerta; el dominio resuelve
# igual por A (198.118.194.34) y desde una red con IPv6 esto no cambia nada.
urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET

# bbox Iberia + Baleares/Canarias fuera (igual que en el modelo)
URL = ("https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
       "{key}/{sat}/-10,35,5,44/{dias}")


class FirmsCaido(RuntimeError):
    """FIRMS no ha devuelto CSV tras agotar los reintentos."""


def descargar(key, sat, dias, intentos=4, espera=30, timeout=60):
    """CSV de FIRMS como DataFrame. Reintenta `intentos` veces con backoff
    lineal (0, `espera`, 2·`espera`... segundos) y levanta FirmsCaido si
    ninguna respuesta es un CSV válido.

    4×30 s = ~3 min de ventana: el 2026-08-07 la API se volvió inalcanzable
    (`Network is unreachable`) entre dos pasadas que sí funcionaron, y con
    una ventana de 1 min el job moría en una caída de nada. Los jobs que
    llaman aquí tienen 30 y 90 min de timeout, sobra margen."""
    ultimo = "sin intentos"
    for i in range(intentos):
        if i:
            time.sleep(espera * i)
        try:
            r = requests.get(URL.format(key=key, sat=sat, dias=dias),
                             timeout=timeout)
        except Exception as e:
            ultimo = f"{type(e).__name__}: {e}"
        else:
            if r.ok and r.text.startswith("latitude"):
                return pd.read_csv(StringIO(r.text))
            ultimo = f"HTTP {r.status_code}: {r.text[:120].strip()}"
        print(f"FIRMS {sat}: intento {i + 1}/{intentos} fallido — {ultimo}",
              flush=True)
    raise FirmsCaido(f"{sat}: {intentos} intentos fallidos — {ultimo}")
