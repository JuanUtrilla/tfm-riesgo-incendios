# Lo que queda

> Estado a **07/09/2026**. Este fichero dice qué falta y qué se decidió dejar
> fuera; la historia de cómo se llegó aquí está en [`BITACORA.md`](BITACORA.md)
> y, para el detalle día a día de agosto y septiembre, en el histórico de este
> mismo fichero en git (`git log -p docs/PENDIENTE.md`).

## Lo único que bloquea la entrega: la memoria

1. **La sección de ML de la memoria conjunta** (5 páginas, divulgativa).
   Primer borrador en [`MEMORIA/seccion_ML_divulgativa.md`](MEMORIA/seccion_ML_divulgativa.md);
   pendiente de lectura del autor. El titular es el replay
   (`REPLAY_VEREDICTO.md`), el modelo propuesto el r10, el producto de dos
   capas, y los límites de `LIMITACIONES.md` §3, §9 y §11 van escritos.
2. **La figura del invierno** (percentil del día frente a escala absoluta, el
   mismo día): no existe. Se genera con `dos_27_escala_absoluta.py` sobre un
   día del replay.
3. **Los anexos** (variables, ablaciones, jueces al cierre, calibración,
   incidentes, guía del repo). El material está todo en `docs/`; es recorte.
4. **Cerrar el párrafo de los jueces en vivo** con las cifras del último día
   de temporada, cuando se decida parar la cadena (a mediados de septiembre).

## Decidido, y por qué

| Decisión | Fecha | Razón |
|---|---|---|
| El veredicto es el replay 2025-2026; los jueces en vivo son validación operativa en curso | 06/09 | 250 días contra 15, dos temporadas, prerregistro. Límite escrito en `LIMITACIONES.md` §3 |
| Ventana de datos de 2026 cerrada el **02-sep** | 07/09 | Es donde llega el archivo IFS; extender 4 días no cambia ninguna conclusión |
| El mapa del 21-ago **se excluye** del juez EFFIS | 06/09 | El original no existe en ningún disco; `LIMITACIONES.md` §11 |
| Los 5 días perdidos (27-31/08) no se regeneran | 31/08 | Un mapa regenerado no es un mapa sellado |
| Bug del viento en producción: **sellado**, cuantificado en anexo | 02/09 | Corregirlo a mitad de temporada rompe la comparación |
| Desajuste caja/radio (`LIMITACIONES.md` §8): se documenta, no se corrige | 05/09 | Escala, no orden (1,39×); tocarlo exige reentrenar |
| Fase 4 del «si» (modelo día-nacional directo): **descartada** | 06/09 | No aporta al argumento; hay material de sobra |
| No se reescribe código; copia literal con md5 | 25/08 | `ESTRUCTURA.md` §2 |
| No se hace CI con badge, `demo_muestras.py` ni `requirements.txt` para pip | 06/09 | Solo valen si el repo se comparte para ejecutarlo; para la memoria no |

## Se deja abierto a propósito (no bloquea nada)

- Recalibrar la probabilidad por celda condicionada a día grande: los tres
  intentos fallan por la misma razón (2022 fuera del envolvente); es una
  limitación del planteamiento, no un pendiente (§9).
- Ampliar `auditoria_train_serve.py` a geometrías y ventanas temporales: vale
  más como nota en `LIMITACIONES.md` que como código, cerrada la fase
  experimental.
- Los 2 commits sin pushear del repositorio original `TFM_fuego`.
- Licencia del repositorio: a decidir con el equipo antes de compartirlo fuera.

## No tocar hasta el cierre de la cadena

`dos_riesgo_hoy.py` (`CORTES_PCTL`), el bug del viento, la ventana FIRMS de la
cadena viva y el modelo servido. Todo lo del capítulo 5.6 es evidencia para
decidir, no un cambio de producto.
