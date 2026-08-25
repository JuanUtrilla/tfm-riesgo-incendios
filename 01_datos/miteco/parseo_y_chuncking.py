# COPIA LITERAL de /home/charredgem/Desktop/Master/TFM-RAG/src/miteco_rag/parseo_y_chuncking.py
# (exportada por gh_exportar_estado.py el 21/08/2026 para correr sin ese repo; no editar aquí: editar allí y reexportar)


# ------------------
# IMPORTS   
# ------------------
from pathlib import Path
from pydantic import BaseModel, Field
from datetime import date, datetime, timezone
from typing import Iterable, Literal
from collections import Counter

import pymupdf
import re
import unicodedata
import hashlib
import json

# ------------------
# CONSTANTES
# ------------------
INPUT_DIR = Path('data/raw/miteco')
SNAPSHOTS_PATH = Path("data/processed/fire_snapshots.jsonl")
REPORT_PATH = Path("data/processed/parser_report.json")
LOCATION_PREFIXES = ("localizacion:",)
SUMMARY_PREFIX = "actuaciones de los medios del ministerio" #texto que marca el resumen final del documento, cuando aparece, ya no hay más incendios
PARSER_VERSION = "0.1.0"

# ------------------
# DICCIONARIOS
# ------------------
SPANISH_MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}

FOREIGN_COUNTRIES = {
    "portugal": "PT",
}

COMMUNITY_ALIASES_RAW = {
    "ANDALUCIA": "Andalucía",
    "ARAGON": "Aragón",
    "ASTURIAS": "Asturias",
    "PRINCIPADO DE ASTURIAS": "Asturias",
    "CANTABRIA": "Cantabria",
    "CASTILLA-LA MANCHA": "Castilla-La Mancha",
    "CASTILLA LA MANCHA": "Castilla-La Mancha",
    "CASTILLA Y LEON": "Castilla y León",
    "CATALUÑA": "Cataluña",
    "CATALUNYA": "Cataluña",
    "CEUTA": "Ceuta",
    "C. VALENCIANA": "Comunitat Valenciana",
    "COMUNIDAD VALENCIANA": "Comunitat Valenciana",
    "COMUNITAT VALENCIANA": "Comunitat Valenciana",
    "EXTREMADURA": "Extremadura",
    "GALICIA": "Galicia",
    "ISLAS BALEARES": "Illes Balears",
    "ILLES BALEARS": "Illes Balears",
    "LA RIOJA": "La Rioja",
    "MADRID": "Comunidad de Madrid",
    "COMUNIDAD DE MADRID": "Comunidad de Madrid",
    "MELILLA": "Melilla",
    "MURCIA": "Región de Murcia",
    "REGION DE MURCIA": "Región de Murcia",
    "NAVARRA": "Navarra",
    "COMUNIDAD FORAL DE NAVARRA": "Navarra",
    "PAIS VASCO": "País Vasco",
    "EUSKADI": "País Vasco",
    "CANARIAS": "Canarias",
}



# Cada provincia se asocia también con su comunidad. Se incluyen variantes
# castellanas y cooficiales frecuentes en los documentos.
PROVINCES_RAW = {
    "A CORUÑA": ("A Coruña", "Galicia"),
    "CORUÑA": ("A Coruña", "Galicia"),
    "ALAVA": ("Álava", "País Vasco"),
    "ARABA": ("Álava", "País Vasco"),
    "ALBACETE": ("Albacete", "Castilla-La Mancha"),
    "ALICANTE": ("Alicante", "Comunitat Valenciana"),
    "ALACANT": ("Alicante", "Comunitat Valenciana"),
    "ALMERIA": ("Almería", "Andalucía"),
    "ASTURIAS": ("Asturias", "Asturias"),
    "AVILA": ("Ávila", "Castilla y León"),
    "BADAJOZ": ("Badajoz", "Extremadura"),
    "BARCELONA": ("Barcelona", "Cataluña"),
    "BIZKAIA": ("Bizkaia", "País Vasco"),
    "VIZCAYA": ("Bizkaia", "País Vasco"),
    "BURGOS": ("Burgos", "Castilla y León"),
    "CACERES": ("Cáceres", "Extremadura"),
    "CADIZ": ("Cádiz", "Andalucía"),
    "CANTABRIA": ("Cantabria", "Cantabria"),
    "CASTELLON": ("Castellón", "Comunitat Valenciana"),
    "CASTELLO": ("Castellón", "Comunitat Valenciana"),
    "CEUTA": ("Ceuta", "Ceuta"),
    "CIUDAD REAL": ("Ciudad Real", "Castilla-La Mancha"),
    "CORDOBA": ("Córdoba", "Andalucía"),
    "CUENCA": ("Cuenca", "Castilla-La Mancha"),
    "GIRONA": ("Girona", "Cataluña"),
    "GERONA": ("Girona", "Cataluña"),
    "GRANADA": ("Granada", "Andalucía"),
    "GUADALAJARA": ("Guadalajara", "Castilla-La Mancha"),
    "GIPUZKOA": ("Gipuzkoa", "País Vasco"),
    "GUIPUZCOA": ("Gipuzkoa", "País Vasco"),
    "HUELVA": ("Huelva", "Andalucía"),
    "HUESCA": ("Huesca", "Aragón"),
    "ILLES BALEARS": ("Illes Balears", "Illes Balears"),
    "BALEARES": ("Illes Balears", "Illes Balears"),
    "JAEN": ("Jaén", "Andalucía"),
    "LA RIOJA": ("La Rioja", "La Rioja"),
    "LAS PALMAS": ("Las Palmas", "Canarias"),
    "LEON": ("León", "Castilla y León"),
    "LLEIDA": ("Lleida", "Cataluña"),
    "LERIDA": ("Lleida", "Cataluña"),
    "LUGO": ("Lugo", "Galicia"),
    "MADRID": ("Madrid", "Comunidad de Madrid"),
    "MALAGA": ("Málaga", "Andalucía"),
    "MELILLA": ("Melilla", "Melilla"),
    "MURCIA": ("Murcia", "Región de Murcia"),
    "NAVARRA": ("Navarra", "Navarra"),
    "OURENSE": ("Ourense", "Galicia"),
    "ORENSE": ("Ourense", "Galicia"),
    "PALENCIA": ("Palencia", "Castilla y León"),
    "PONTEVEDRA": ("Pontevedra", "Galicia"),
    "SALAMANCA": ("Salamanca", "Castilla y León"),
    "SANTA CRUZ DE TENERIFE": ("Santa Cruz de Tenerife", "Canarias"),
    "SEGOVIA": ("Segovia", "Castilla y León"),
    "SEVILLA": ("Sevilla", "Andalucía"),
    "SORIA": ("Soria", "Castilla y León"),
    "TARRAGONA": ("Tarragona", "Cataluña"),
    "TERUEL": ("Teruel", "Aragón"),
    "TOLEDO": ("Toledo", "Castilla-La Mancha"),
    "VALENCIA": ("Valencia", "Comunitat Valenciana"),
    "VALENCIA/VALENCIA": ("Valencia", "Comunitat Valenciana"),
    "VALLADOLID": ("Valladolid", "Castilla y León"),
    "ZAMORA": ("Zamora", "Castilla y León"),
    "ZARAGOZA": ("Zaragoza", "Aragón"),
}

# ------------------
# EXPRESIONES REGULARES
# ------------------
TEXTUAL_DATE_PATTERN = re.compile(
    r"\b(\d{1,2})\s+de\s+("
    + "|".join(SPANISH_MONTHS)
    + r")\s+de\s+(\d{4})\b"
)

NUMERIC_DATE_PATTERN = re.compile(
    r"\bfecha\s*:\s*(\d{1,2})/(\d{1,2})/(\d{4})\b"
)

LAST_UPDATE_PATTERN = re.compile(
    r"ultima\s+actualizacion\s*:\s*"
    r"(\d{1,2}:\d{2}(?::\d{2})?)\s*"
    r"(?:del\s+dia\s*)?"
    r"(\d{1,2})/(\d{1,2})/(\d{4})"
)

STATUS_PATTERN = re.compile(
    r"estado\s+del\s+incendio\s*:\s*"
    r"([A-ZÁÉÍÓÚÜÑ]+)"
    r"(?:\s+S\.O\s*:\s*([A-Z0-9]+))?",
    flags=re.IGNORECASE,
)

INCIDENT_START_PATTERN = re.compile(
    r"incendio\s+iniciado\s+el\s+"
    r"(\d{1,2})/(\d{1,2})/(\d{4})?",
    flags=re.IGNORECASE,
)

RESOURCE_PATTERN = re.compile(
    r"^(\d+)\s+([A-ZÁÉÍÓÚÜÑ]+(?:-[A-ZÁÉÍÓÚÜÑ]+)?)\b\s*(.*)$"
)

# ------------------
# CLASES
# ------------------
class PDFLine(BaseModel):
    page_number: int
    line_number: int
    raw_text: str
    cleaned_text: str
    normalized_text: str

class DocumentMetadata(BaseModel):
    """Datos comunes a todos los incendios de un PDF."""
    document_id: str
    source_file: str
    source_path: str
    source_sha256: str
    source_url: str | None = None
    report_type: Literal["definitivo", "provisional", "desconocido"]
    report_date: date
    last_update: datetime | None = None

class FireBlock(BaseModel):
    """Grupo de líneas que ya sabemos que pertenecen al mismo incendio."""
    ordinal: int = Field(ge=1)
    country: str
    autonomous_community: str | None
    province: str | None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    lines: list[PDFLine]

class AssignedResource(BaseModel):
    """Medio asignado extraído del bloque del incendio."""
    raw_text: str
    quantity: int | None = Field(default=None, ge=1)
    code: str | None = None
    description: str | None = None
    origin: str | None = None

class FireSnapshot(BaseModel):
    """Estado de un incendio en un parte diario concreto."""
    snapshot_id: str
    incident_key: str
    document_id: str

    country: str
    autonomous_community: str | None
    autonomous_community_normalized: str | None
    province: str | None
    province_normalized: str | None
    location: str
    location_normalized: str

    status: str | None
    operational_status: str | None
    note: str | None
    incident_start_date: date | None
    assigned_resources: list[AssignedResource]
    resource_codes: list[str]

    report_date: date
    report_date_number: int
    last_update: datetime | None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    source_file: str
    source_url: str | None
    source_sha256: str
    parser_version: str = PARSER_VERSION

    raw_text: str = Field(min_length=1)
    chunk_text: str = Field(min_length=1)

class ParserReport(BaseModel):
    """Informe de calidad y recuentos de la ejecución."""
    parser_version: str = PARSER_VERSION
    generated_at: datetime
    processed_files: list[str]
    snapshots_by_file: dict[str, int]
    total_snapshots: int = Field(ge=0)
    spanish_snapshots: int = Field(ge=0)
    foreign_snapshots: int = Field(ge=0)
    warnings: list[str]
    errors: list[str]

# ------------------
# FUNCIONES
# ------------------
def clean_line(line: str) -> str:
    '''
    Elimina espacios sobrantes
    \s+ significa uno o más espacios, tabuladores o saltos internos.
    re.sub() pertenece al módulo re (expresiones regulares)
    Sintaxis: re.sub(patrón, reemplazo, cadena)
    .strip() elimina espacios al inicio y al final de la cadena
    '''
    return re.sub(r'\s+', ' ', line).strip() 

def normalize_text(text: str | None) -> str:
    '''
    Normaliza texto eliminando acentos y convirtiendo a minúsculas
    '''
    if not text:
        return ''
    
    decomposed = unicodedata.normalize('NFKD', text) #NFKD separa p.e. "á" en "a" y "´"
    # Eliminamos los carácteres que son marcas diacríticas
    without_accents = ''.join(
        character
        for character in decomposed
        if not unicodedata.combining(character) # True si el carácter es una marca diacrítica
    )

    return clean_line(without_accents).lower()

AUTONOMOUS_COMMUNITIES = {
    normalize_text(alias): canonical
    for alias, canonical in COMMUNITY_ALIASES_RAW.items()
}

PROVINCE_TO_COMMUNITY = {
    normalize_text(alias): value
    for alias, value in PROVINCES_RAW.items()
}



def calculate_sha256(path: Path, buffer_size: int = 1024 * 1024) -> str:
    """Calcula el hash por bloques para no cargar todo el PDF en memoria."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(buffer_size):
            digest.update(block)

    return digest.hexdigest()

def extract_pdf_lines(pdf_path: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"El archivo {pdf_path} no existe.")
    
    extracted_lines = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text('text')
            for line_number, raw_line in enumerate(page_text.splitlines(), start=1):
                cleaned_line = clean_line(raw_line)
                if not cleaned_line:
                    continue

                extracted_lines.append(
                    PDFLine(
                        page_number=page_number,
                        line_number=line_number,
                        raw_text=raw_line,
                        cleaned_text=cleaned_line,
                        normalized_text=normalize_text(cleaned_line)
                    )
                )
    
    if not extracted_lines:
        raise ValueError(f'El PDF no contiene texto extraíble: {pdf_path.name}')
    
    return extracted_lines

def extract_report_date(text: str) -> date:
    '''
    Busca la fecha en el texto del informe y la devuelve como un objeto date.
    Si no se encuentra una fecha válida, lanza un ValueError.
    '''
    textual_match = TEXTUAL_DATE_PATTERN.search(text)
    if textual_match:
        day = int(textual_match.group(1))
        month = SPANISH_MONTHS[textual_match.group(2).lower()]
        year = int(textual_match.group(3))
        return date(year, month, day)
    
    numeric_match = NUMERIC_DATE_PATTERN.search(text)
    if numeric_match:
        day = int(numeric_match.group(1))
        month = int(numeric_match.group(2))
        year = int(numeric_match.group(3))
        return date(year, month, day)
    
    raise ValueError("No se pudo extraer la fecha del informe.")

def extract_last_update(text: str) -> datetime | None:
    '''
    Busca la última actualización en el texto del informe y la devuelve como un objeto datetime.
    Si no se encuentra una fecha válida, devuelve None.
    '''
    match = LAST_UPDATE_PATTERN.search(text)
    if match:
        time_text = match.group(1)
        day = int(match.group(2))
        month = int(match.group(3))
        year = int(match.group(4))

        if time_text.count(':') == 1:
            time_text += ':00'  # Añadir segundos si no están presentes
        
        return datetime.strptime(
            f"{year}-{month}-{day} {time_text}",
            "%Y-%m-%d %H:%M:%S",
        )
        
    return None

def infer_report_type(pdf_path: Path) -> Literal["definitivo", "provisional", "desconocido"]:
    '''Infiere el tipo únicamente cuando el nombre es inequívoco.'''

    normalized_name = normalize_text(pdf_path.stem) #.stem elimina la extensión (p.e. '.pdf')

    if "definitivo" in normalized_name:
        return "definitivo"
    if "provisional" in normalized_name:
        return "provisional"
    return "desconocido"

def extract_document_metadata(
    pdf_path: Path,
    lines: list[PDFLine],
    source_url: str | None = None,
) -> DocumentMetadata:
    """Construye los metadatos compartidos por todos los bloques."""

    normalized_document = "\n".join(line.normalized_text for line in lines)
    sha256 = calculate_sha256(pdf_path)

    return DocumentMetadata(
        document_id=sha256[:20],
        source_file=pdf_path.name,
        source_path=str(pdf_path),
        source_sha256=sha256,
        source_url=source_url,
        report_type=infer_report_type(pdf_path),
        report_date=extract_report_date(normalized_document),
        last_update=extract_last_update(normalized_document),
    )

def is_location_start(line: PDFLine) -> bool:
    """
    La normalización ya ha eliminado el acento de Localización.
    Devuelve True si la línea comienza con "localizacion:".
    """

    return line.normalized_text.startswith(LOCATION_PREFIXES)

def is_summary_start(line: PDFLine) -> bool:
    """
    Marca el límite final de los bloques de incendios.
    Devuelve True si la línea comienza con "actuaciones de los medios del ministerio".
    """

    return line.normalized_text.startswith(SUMMARY_PREFIX)

def split_fire_blocks(lines: list[PDFLine]) -> list[FireBlock]:
    """Delimita un bloque por incendio conservando la geografía vigente."""

    blocks: list[FireBlock] = []

    current_country = "ES"
    current_community: str | None = None
    current_province: str | None = None

    # Estos valores solo existen mientras hay un incendio abierto.
    current_lines: list[PDFLine] = []
    block_country = "ES"
    block_community: str | None = None
    block_province: str | None = None

    def close_current_block() -> None:
        """Cierra el bloque abierto utilizando una copia de su estado."""

        nonlocal current_lines

        if not current_lines:
            return

        blocks.append(
            FireBlock(
                ordinal=len(blocks) + 1,
                country=block_country,
                autonomous_community=block_community,
                province=block_province,
                page_start=current_lines[0].page_number,
                page_end=current_lines[-1].page_number,
                lines=list(current_lines),
            )
        )
        current_lines = []

    for line in lines:
        normalized = line.normalized_text

        # El resumen estadístico no pertenece al último incendio.
        if is_summary_start(line):
            close_current_block()
            break

        # "OTRO PAIS" anuncia que el encabezado siguiente no es español.
        if normalized == "otro pais":
            close_current_block()
            current_country = "OTHER"
            current_community = None
            current_province = None
            continue

        # Después de OTRO PAIS reconocemos el nombre del país.
        if current_country != "ES" and normalized in FOREIGN_COUNTRIES:
            close_current_block()
            current_country = FOREIGN_COUNTRIES[normalized]
            current_community = None
            current_province = line.cleaned_text
            continue

        province_candidate = PROVINCE_TO_COMMUNITY.get(normalized)
        community_candidate = AUTONOMOUS_COMMUNITIES.get(normalized)

        # Asturias, Madrid, Murcia, Navarra y La Rioja pueden ser a la vez
        # comunidad y provincia. Si la comunidad ya está activa, la segunda
        # aparición se interpreta como provincia.
        if (
            province_candidate
            and current_community == province_candidate[1]
        ):
            close_current_block()
            current_country = "ES"
            current_province = province_candidate[0]
            continue

        if community_candidate:
            close_current_block()
            current_country = "ES"
            current_community = community_candidate
            current_province = None
            continue

        if province_candidate:
            close_current_block()
            current_country = "ES"
            current_province, current_community = province_candidate
            continue

        if is_location_start(line):
            # Una nueva Localización siempre cierra el incendio anterior.
            close_current_block()

            # Congelamos la geografía para que cambios posteriores no alteren
            # retroactivamente este bloque.
            block_country = current_country
            block_community = current_community
            block_province = current_province
            current_lines = [line]
            continue

        # Solo guardamos contenido cuando ya se ha abierto un incendio.
        if current_lines:
            current_lines.append(line)

    # Seguridad para documentos que terminen sin marcador de resumen.
    close_current_block()

    if not blocks:
        raise ValueError("No se encontraron bloques iniciados por Localización:")

    return blocks

def block_text(block: FireBlock) -> str:
    """Une las líneas de un bloque sin perder sus límites semánticos."""

    return "\n".join(line.cleaned_text for line in block.lines)


def extract_location(block: FireBlock) -> str:
    """Obtiene el valor situado después del primer signo de dos puntos."""

    first_line = block.lines[0].cleaned_text

    if ":" not in first_line:
        raise ValueError(f"Bloque {block.ordinal} sin localización válida")

    location = clean_line(first_line.split(":", maxsplit=1)[1])

    if not location:
        raise ValueError(f"Bloque {block.ordinal} con localización vacía")

    return location


def extract_status(block: FireBlock) -> tuple[str | None, str | None]:
    """Devuelve estado y situación operativa como valores separados."""

    match = STATUS_PATTERN.search(block_text(block))
    if not match:
        return None, None

    status = match.group(1).upper()
    operational_status = (
        match.group(2).upper()
        if match.group(2)
        else None
    )
    return status, operational_status


def extract_note(
    block: FireBlock,
) -> tuple[str | None, date | None]:
    """Extrae la nota y una fecha de inicio solo cuando incluye año."""

    note_start: int | None = None

    for index, line in enumerate(block.lines):
        if line.normalized_text.startswith("nota:"):
            note_start = index
            break

    if note_start is None:
        return None, None

    first_note_line = block.lines[note_start].cleaned_text.split(
        ":", maxsplit=1
    )[1]
    continuation = [
        line.cleaned_text
        for line in block.lines[note_start + 1 :]
    ]
    note = clean_line(" ".join([first_note_line, *continuation]))

    start_match = INCIDENT_START_PATTERN.search(note)
    if not start_match or not start_match.group(3):
        # Si falta el año conservamos la nota, pero no inventamos la fecha.
        return note, None

    day, month, year = map(int, start_match.groups())
    return note, date(year, month, day)


def extract_assigned_resources(
    block: FireBlock,
) -> tuple[list[AssignedResource], list[str]]:
    """Extrae medios sin exigir una estructura perfecta en la primera fase."""

    resources: list[AssignedResource] = []
    inside_resources = False

    for line in block.lines:
        normalized = line.normalized_text

        if normalized.startswith("medios asignados por el miteco"):
            inside_resources = True
            continue

        if not inside_resources:
            continue

        # La línea de estado no es un medio.
        if normalized.startswith("estado del incendio"):
            continue

        # La nota marca el final de la lista de medios.
        if normalized.startswith("nota:"):
            break

        match = RESOURCE_PATTERN.match(line.cleaned_text)

        if match:
            quantity = int(match.group(1))
            code = match.group(2).upper()
            description = clean_line(match.group(3)) or None

            resources.append(
                AssignedResource(
                    raw_text=line.cleaned_text,
                    quantity=quantity,
                    code=code,
                    description=description,
                    origin=None,
                )
            )
        elif resources:
            # Una línea sin cantidad continúa la descripción anterior.
            previous = resources[-1]
            previous.raw_text = clean_line(
                f"{previous.raw_text} {line.cleaned_text}"
            )
            previous.description = clean_line(
                f"{previous.description or ''} {line.cleaned_text}"
            )

    # dict.fromkeys elimina duplicados conservando el orden.
    resource_codes = list(
        dict.fromkeys(
            resource.code
            for resource in resources
            if resource.code
        )
    )

    return resources, resource_codes

def short_sha256(value: str, length: int = 20) -> str:
    """Hash corto y determinista para identificadores internos."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def build_snapshot_id(
    document: DocumentMetadata,
    block: FireBlock,
    location_normalized: str,
) -> str:
    """Identifica de forma única el snapshot dentro del documento."""

    raw_identifier = (
        f"{document.source_sha256}|"
        f"{block.ordinal}|"
        f"{location_normalized}"
    )
    return short_sha256(raw_identifier)


def build_incident_key(
    country: str,
    community_normalized: str | None,
    province_normalized: str | None,
    location_normalized: str,
    incident_start_date: date | None,
) -> str:
    """Clave estable que no incluye estado, medios ni texto cambiante."""

    raw_identifier = "|".join(
        [
            country,
            community_normalized or "",
            province_normalized or "",
            location_normalized,
            incident_start_date.isoformat() if incident_start_date else "",
        ]
    )
    return short_sha256(raw_identifier)


def build_chunk_text(
    document: DocumentMetadata,
    block: FireBlock,
    location: str,
    status: str | None,
    operational_status: str | None,
    resources: list[AssignedResource],
    note: str | None,
) -> str:
    """Genera una representación autosuficiente para el futuro embedding."""

    parts = [
        f"Fecha del parte: {document.report_date.isoformat()}",
        f"País: {block.country}",
        (
            f"Comunidad autónoma: {block.autonomous_community}"
            if block.autonomous_community
            else None
        ),
        f"Provincia: {block.province}" if block.province else None,
        f"Localización: {location}",
        f"Estado: {status}" if status else None,
        (
            f"Situación operativa: {operational_status}"
            if operational_status
            else None
        ),
        "Medios asignados:",
        *[f"- {resource.raw_text}" for resource in resources],
        f"Nota: {note}" if note else None,
        f"Fuente: {document.source_file}, página {block.page_start}",
    ]

    return "\n".join(part for part in parts if part is not None)


def build_fire_snapshot(
    document: DocumentMetadata,
    block: FireBlock,
) -> FireSnapshot:
    """Coordina los extractores y deja que Pydantic valide el resultado."""

    location = extract_location(block)
    location_normalized = normalize_text(location)

    status, operational_status = extract_status(block)
    note, incident_start_date = extract_note(block)
    resources, resource_codes = extract_assigned_resources(block)

    community_normalized = (
        normalize_text(block.autonomous_community)
        if block.autonomous_community
        else None
    )
    province_normalized = (
        normalize_text(block.province)
        if block.province
        else None
    )

    return FireSnapshot(
        snapshot_id=build_snapshot_id(
            document,
            block,
            location_normalized,
        ),
        incident_key=build_incident_key(
            country=block.country,
            community_normalized=community_normalized,
            province_normalized=province_normalized,
            location_normalized=location_normalized,
            incident_start_date=incident_start_date,
        ),
        document_id=document.document_id,
        country=block.country,
        autonomous_community=block.autonomous_community,
        autonomous_community_normalized=community_normalized,
        province=block.province,
        province_normalized=province_normalized,
        location=location,
        location_normalized=location_normalized,
        status=status,
        operational_status=operational_status,
        note=note,
        incident_start_date=incident_start_date,
        assigned_resources=resources,
        resource_codes=resource_codes,
        report_date=document.report_date,
        report_date_number=int(document.report_date.strftime("%Y%m%d")),
        last_update=document.last_update,
        page_start=block.page_start,
        page_end=block.page_end,
        source_file=document.source_file,
        source_url=document.source_url,
        source_sha256=document.source_sha256,
        parser_version=PARSER_VERSION,
        raw_text=block_text(block),
        chunk_text=build_chunk_text(
            document=document,
            block=block,
            location=location,
            status=status,
            operational_status=operational_status,
            resources=resources,
            note=note,
        ),
    )

def parse_miteco_pdf(
    pdf_path: Path,
    source_url: str | None = None,
) -> list[FireSnapshot]:
    """Ejecuta la fase completa sobre un único PDF."""

    lines = extract_pdf_lines(pdf_path)
    document = extract_document_metadata(
        pdf_path,
        lines,
        source_url=source_url,
    )
    blocks = split_fire_blocks(lines)

    return [
        build_fire_snapshot(document, block)
        for block in blocks
    ]


def parse_pdf_directory(input_dir: Path) -> list[FireSnapshot]:
    """Parsea todos los PDF sin deduplicar snapshots de días distintos."""

    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No se encontraron PDF en {input_dir.resolve()}"
        )

    snapshots: list[FireSnapshot] = []

    for pdf_path in pdf_files:
        snapshots.extend(parse_miteco_pdf(pdf_path))

    return snapshots

def validate_snapshots(
    snapshots: list[FireSnapshot],
) -> tuple[list[str], list[str]]:
    """Comprueba unicidad, geografía, páginas y contaminación del resumen."""

    warnings: list[str] = []
    errors: list[str] = []

    snapshot_ids = [snapshot.snapshot_id for snapshot in snapshots]
    duplicated_ids = [
        identifier
        for identifier, count in Counter(snapshot_ids).items()
        if count > 1
    ]

    if duplicated_ids:
        errors.append(
            f"snapshot_id duplicados: {duplicated_ids}"
        )

    for snapshot in snapshots:
        label = f"{snapshot.source_file}:{snapshot.location}"

        if snapshot.page_start > snapshot.page_end:
            errors.append(f"{label}: rango de páginas inválido")

        if SUMMARY_PREFIX in normalize_text(snapshot.raw_text):
            errors.append(f"{label}: contiene el resumen estadístico")

        if snapshot.country == "ES" and not snapshot.province:
            warnings.append(f"{label}: provincia española ausente")

        if not snapshot.resource_codes:
            warnings.append(f"{label}: no se detectaron códigos de medios")

    return warnings, errors


def create_parser_report(
    snapshots: list[FireSnapshot],
    processed_files: Iterable[Path],
    warnings: list[str],
    errors: list[str],
) -> ParserReport:
    """Resume recuentos por documento y país."""

    snapshots_by_file = dict(
        sorted(Counter(
            snapshot.source_file
            for snapshot in snapshots
        ).items())
    )

    return ParserReport(
        generated_at=datetime.now(timezone.utc),
        processed_files=[
            path.name
            for path in sorted(processed_files)
        ],
        snapshots_by_file=snapshots_by_file,
        total_snapshots=len(snapshots),
        spanish_snapshots=sum(
            snapshot.country == "ES"
            for snapshot in snapshots
        ),
        foreign_snapshots=sum(
            snapshot.country != "ES"
            for snapshot in snapshots
        ),
        warnings=warnings,
        errors=errors,
    )


def write_snapshots_jsonl(
    snapshots: Iterable[FireSnapshot],
    output_path: Path,
) -> None:
    """Escribe un objeto JSON por línea."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for snapshot in snapshots:
            serialized = snapshot.model_dump(mode="json")
            file.write(
                json.dumps(serialized, ensure_ascii=False) + "\n"
            )


def write_parser_report(
    report: ParserReport,
    output_path: Path,
) -> None:
    """Guarda el informe con indentación para revisión humana."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            report.model_dump(mode="json"),
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")


def run_phase1(
    input_dir: Path,
    snapshots_path: Path,
    report_path: Path,
) -> ParserReport:
    """Parsea, valida y exporta el corpus completo de la fase 1."""

    pdf_files = sorted(input_dir.glob("*.pdf"))
    snapshots = parse_pdf_directory(input_dir)
    warnings, errors = validate_snapshots(snapshots)
    report = create_parser_report(
        snapshots=snapshots,
        processed_files=pdf_files,
        warnings=warnings,
        errors=errors,
    )

    if errors:
        raise RuntimeError(
            "No se exporta porque el parser contiene errores: "
            + "; ".join(errors)
        )

    write_snapshots_jsonl(snapshots, snapshots_path)
    write_parser_report(report, report_path)
    return report


def main() -> None:
    """Ejecuta la fase 1 usando las rutas configuradas del proyecto."""

    report = run_phase1(
        input_dir=INPUT_DIR,
        snapshots_path=SNAPSHOTS_PATH,
        report_path=REPORT_PATH,
    )

    print(json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    ))
    print("Snapshots:", SNAPSHOTS_PATH.resolve())
    print("Informe:", REPORT_PATH.resolve())


if __name__ == "__main__":
    main()


