#!/usr/bin/env python3
"""
Sistema canadiense FWI (Van Wagner 1987) — implementación numpy pura.

Calcula FFMC, DMC, DC, ISI, BUI y FWI a partir de series diarias de:
  temp (°C, aprox. tmax), rh (%, aprox. hrMin), viento (km/h) y precip (mm/24h).

Aproximación asumida (documentada en MODELO_B_BITACORA §14): el sistema canónico
usa valores a mediodía; usamos tmax/hrMin como proxy (práctica común cuando no
hay observación de mediodía). El sesgo resultante se cancela en el percentil
local porque la climatología de referencia se calcula con ESTE mismo módulo.

Ejecutado como script: valida la implementación contra el FWI del cubo IberFire
(misma meteo de entrada → correlación y sesgo por celda).

Referencia: Van Wagner, C.E. 1987. Development and structure of the Canadian
Forest Fire Weather Index System. For. Tech. Rep. 35. (ecuaciones estándar)
"""

import numpy as np

DIAS_MES_EL = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
DIAS_MES_FL = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]


def _ffmc(t, rh, w, p, ffmc0):
    mo = 147.2 * (101.0 - ffmc0) / (59.5 + ffmc0)
    if p > 0.5:
        rf = p - 0.5
        if mo > 150.0:
            mo = (mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo))
                  * (1.0 - np.exp(-6.93 / rf))
                  + 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf))
        else:
            mo = (mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo))
                  * (1.0 - np.exp(-6.93 / rf)))
        mo = min(mo, 250.0)
    ed = (0.942 * rh ** 0.679 + 11.0 * np.exp((rh - 100.0) / 10.0)
          + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * rh)))
    if mo > ed:
        ko = (0.424 * (1.0 - (rh / 100.0) ** 1.7)
              + 0.0694 * np.sqrt(w) * (1.0 - (rh / 100.0) ** 8))
        kd = ko * 0.581 * np.exp(0.0365 * t)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    else:
        ew = (0.618 * rh ** 0.753 + 10.0 * np.exp((rh - 100.0) / 10.0)
              + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * rh)))
        if mo < ew:
            k1 = (0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7)
                  + 0.0694 * np.sqrt(w) * (1.0 - ((100.0 - rh) / 100.0) ** 8))
            kw = k1 * 0.581 * np.exp(0.0365 * t)
            m = ew - (ew - mo) * 10.0 ** (-kw)
        else:
            m = mo
    return 59.5 * (250.0 - m) / (147.2 + m)


def _dmc(t, rh, p, dmc0, mes):
    if p > 1.5:
        re = 0.92 * p - 1.27
        mo = 20.0 + np.exp(5.6348 - dmc0 / 43.43)
        if dmc0 <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc0)
        elif dmc0 <= 65.0:
            b = 14.0 - 1.3 * np.log(dmc0)
        else:
            b = 6.2 * np.log(dmc0) - 17.2
        mr = mo + 1000.0 * re / (48.77 + b * re)
        dmc0 = max(244.72 - 43.43 * np.log(mr - 20.0), 0.0)
    if t > -1.1:
        k = 1.894 * (t + 1.1) * (100.0 - rh) * DIAS_MES_EL[mes - 1] * 1e-6
    else:
        k = 0.0
    return dmc0 + 100.0 * k


def _dc(t, p, dc0, mes):
    if p > 2.8:
        rd = 0.83 * p - 1.27
        qo = 800.0 * np.exp(-dc0 / 400.0)
        qr = qo + 3.937 * rd
        dc0 = max(400.0 * np.log(800.0 / qr), 0.0)
    if t > -2.8:
        v = 0.36 * (t + 2.8) + DIAS_MES_FL[mes - 1]
    else:
        v = DIAS_MES_FL[mes - 1]
    return dc0 + max(0.5 * v, 0.0)


def _isi(w, ffmc):
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    ff = 91.9 * np.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)
    return 0.208 * np.exp(0.05039 * w) * ff


def _bui(dmc, dc):
    if dmc <= 0.4 * dc:
        b = 0.8 * dmc * dc / (dmc + 0.4 * dc) if (dmc + 0.4 * dc) > 0 else 0.0
    else:
        b = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
    return max(b, 0.0)


def _fwi(isi, bui):
    if bui <= 80.0:
        fd = 0.626 * bui ** 0.809 + 2.0
    else:
        fd = 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui))
    b = 0.1 * isi * fd
    if b > 1.0:
        return np.exp(2.72 * (0.434 * np.log(b)) ** 0.647)
    return b


def calcular_fwi_serie(temp, rh, viento_kmh, precip, meses,
                       ffmc0=85.0, dmc0=6.0, dc0=15.0):
    """Serie FWI completa. Entradas: arrays diarios alineados; NaN → se
    propaga el estado anterior (día sin dato no actualiza los códigos).
    Devuelve dict de arrays: ffmc, dmc, dc, isi, bui, fwi."""
    n = len(temp)
    out = {k: np.full(n, np.nan) for k in ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]}
    ffmc, dmc, dc = ffmc0, dmc0, dc0
    for i in range(n):
        t, h, w, p, mes = temp[i], rh[i], viento_kmh[i], precip[i], int(meses[i])
        if np.isnan(t) or np.isnan(h):
            continue
        w = 0.0 if np.isnan(w) else w
        p = 0.0 if np.isnan(p) else p
        h = min(h, 100.0)
        ffmc = _ffmc(t, h, w, p, ffmc)
        dmc = _dmc(t, h, p, dmc, mes)
        dc = _dc(t, p, dc, mes)
        isi = _isi(w, ffmc)
        bui = _bui(dmc, dc)
        out["ffmc"][i], out["dmc"][i], out["dc"][i] = ffmc, dmc, dc
        out["isi"][i], out["bui"][i], out["fwi"][i] = isi, bui, _fwi(isi, bui)
    return out


def _validar_contra_cubo(n_celdas=60, anio=2014):
    """Corre este FWI sobre la meteo del cubo y lo compara con el FWI del cubo."""
    import xarray as xr
    import config
    ds = xr.open_dataset(config.CUBO, decode_timedelta=False)
    tiempos = ds["time"].values.astype("datetime64[D]")
    # spin-up desde el año anterior para converger DC
    m = (tiempos >= np.datetime64(f"{anio-1}-01-01")) & \
        (tiempos <= np.datetime64(f"{anio}-12-31"))
    idx = np.where(m)[0]
    meses = (tiempos[m].astype("datetime64[M]").astype(int) % 12) + 1
    eval_m = tiempos[m] >= np.datetime64(f"{anio}-06-01")

    rng = np.random.default_rng(1)
    es_esp = ds["is_spain"].values.astype(bool)
    yy, xx = np.where(es_esp)
    sel = rng.choice(len(yy), n_celdas, replace=False)
    cors, sesgos = [], []
    for k in sel:
        iy, ix = int(yy[k]), int(xx[k])
        t = ds["t2m_max"].isel(y=iy, x=ix, time=idx).values
        h = ds["RH_min"].isel(y=iy, x=ix, time=idx).values
        w = ds["wind_speed_max"].isel(y=iy, x=ix, time=idx).values * 3.6
        p = ds["total_precipitation_mean"].isel(y=iy, x=ix, time=idx).values
        ref = ds["FWI"].isel(y=iy, x=ix, time=idx).values
        mio = calcular_fwi_serie(t, h, w, p, meses)["fwi"]
        ok = eval_m & ~np.isnan(mio) & ~np.isnan(ref)
        if ok.sum() < 100:
            continue
        cors.append(np.corrcoef(mio[ok], ref[ok])[0, 1])
        sesgos.append(np.mean(mio[ok] - ref[ok]))
    print(f"Validación vs FWI del cubo ({len(cors)} celdas, jun-dic {anio}):")
    print(f"  correlación: mediana {np.median(cors):.3f} "
          f"(p10 {np.percentile(cors,10):.3f})")
    print(f"  sesgo medio (mío−cubo): mediana {np.median(sesgos):+.1f} "
          f"(p10 {np.percentile(sesgos,10):+.1f}, p90 {np.percentile(sesgos,90):+.1f})")
    print("  (el sesgo se cancela en el percentil local: misma implementación "
          "en serie y climatología)")


if __name__ == "__main__":
    _validar_contra_cubo()
