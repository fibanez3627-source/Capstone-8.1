'''
Generador de fichas técnicas (datasheets) en PDF con xhtml2pdf.

Dos formas de usar la librería:

1) STANDALONE (como la versión original): busca el modelo en una carpeta de CSV / Excel
   (.csv, .xlsx, .xlsm, .xls; en Excel se leen todas las hojas).

    generador = DatasheetGenerator.desde_archivo_json("config.json")
    archivo_generado = generador.generar_pdf("AX-4192401")      # -> ruta del PDF

    config.json
    {
      "rutas": {
        "datos_csv": "datos_csv",
        "template_html": "templates/template.html",
        "fotos": "fotos",
        "logos": "logos",
        "salidas_pdf": "salidas_pdf",
        "vistas_explosionadas": "vistas_explosionadas"
      },
      "mapeo_categorias": {
        "valvulas": {"color": "#00539C"},
        "bombas": {"color": "#2E7D32"},
        "default": {"color": "#424242"}
      }
    }

2) DENTRO DEL AGENTE RELANT (app/products.py): los datos ya vienen del índice de
   productos (CSV de RAG_DIR_PRODUCTOS, fotos de RAG_DIR_FOTOS, vistas de
   RAG_DIR_EXPLOSIONADOS), así que no se busca nada: se compila directo a bytes.

    pdf = DatasheetGenerator(crear_carpetas=False).generar_pdf_bytes(
        modelo="BP-200", categoria="Bombas", variantes=[[("Presión", "220 bar"), ...]],
        ruta_foto=".../BP-200.jpg", ruta_vista_explosionada=None, ruta_logo=".../logo.png")

Variables disponibles en la plantilla HTML (string.Template, $Variable):
    $Modelo $Nombre $Categoria $Color_Acento $Fecha $Tabla_Especificaciones
    $Ruta_Logo $Ruta_Foto $Ruta_Vista_Explosionada
    $Foto_W $Foto_H $Vista_W $Vista_H $Logo_H          (tamaños en px, opcionales)
    $Bloque_Vista_Explosionada   (HTML de la vista explosionada, vacío si no existe)
Si no existe la plantilla configurada se usa PLANTILLA_DEFAULT (formato carta).
'''
from __future__ import annotations

import base64
import glob
import html
import io
import json
import logging
import os
import unicodedata
from datetime import date
from string import Template
from typing import Iterable, Sequence

log = logging.getLogger("rag.pdfdatagen")

# 1x1 px blanco: se usa cuando falta una imagen (xhtml2pdf falla con src vacío)
_PIXEL_BLANCO = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
                 "2mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==")

Variante = Sequence[tuple[str, str]]          # [(característica, valor), ...]


def _clave(s: str) -> str:
    """'Válvulas atomizado' -> 'valvulas atomizado' (para el mapeo de colores)."""
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace("_", " ").replace("-", " ").split())


PLANTILLA_DEFAULT = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Ficha técnica $Modelo</title>
<style>
  @page { size: letter; margin: 14mm 16mm 20mm 16mm;
          @frame footer { -pdf-frame-content: pie; left: 16mm; right: 16mm; bottom: 8mm; height: 10mm; } }
  body { font-family: Helvetica; font-size: 9.5pt; color: #1d1d1b; }
  .barra { background-color: $Color_Acento; height: 4px; }
  h1 { font-size: 16pt; margin: 4mm 0 1mm 0; text-align: center; }
  .pn { text-align: center; font-size: 11pt; font-weight: bold; margin: 0 0 6mm 0; }
  .cat { color: #555555; font-weight: normal; }
  table.specs { width: 100%; }
  table.specs th { background-color: $Color_Acento; color: #ffffff; padding: 4px 5px; text-align: left;
                   border: 0.5px solid $Color_Acento; }
  table.specs td { border: 0.5px solid #9aa0a6; padding: 3px 5px; vertical-align: top; }
  table.specs tr.par td { background-color: #f4f5f6; }
  .caja { border: 0.5px solid #c8ccd0; padding: 4px; text-align: center; }
  .cap { text-align: center; font-size: 8pt; color: #555555; padding-top: 2px; }
  #pie { font-size: 7.5pt; color: #6b7178; border-top: 0.5px solid #c8ccd0; padding-top: 2mm; }
  .rojo { color: $Color_Acento; font-weight: bold; }
</style></head>
<body>
  <table width="100%"><tr>
    <td style="vertical-align: middle;"><span class="rojo" style="font-size: 12pt;">Ficha técnica</span></td>
    <td style="text-align: right;"><img src="$Ruta_Logo" height="$Logo_H"></td>
  </tr></table>
  <div class="barra"></div>
  <h1>$Nombre</h1>
  <p class="pn">$Modelo <span class="cat">· $Categoria</span></p>
  <table width="100%"><tr>
    <td width="58%" style="vertical-align: top; padding-right: 6mm;">$Tabla_Especificaciones</td>
    <td width="42%" style="vertical-align: top;">
      <div class="caja"><img src="$Ruta_Foto" width="$Foto_W" height="$Foto_H"></div>
      <div class="cap">Imagen del producto</div>
      $Bloque_Vista_Explosionada
    </td>
  </tr></table>
  <div id="pie"><span class="rojo">Relant</span> · Ficha técnica generada el $Fecha · $Modelo</div>
</body></html>
"""


class DatasheetGenerator:
    """
    Clase encargada de buscar productos en bases de datos CSV y
    compilar fichas técnicas personalizadas en formato PDF.
    """

    # Mapeo de personalización estética por defecto según la categoría
    MAPEO_CATEGORIAS_DEFAULT = {
        "valvulas": {"color": "#C62828"},
        "bombas": {"color": "#C62828"},
        "actuadores": {"color": "#C62828"},
        "default": {"color": "#C62828"},
    }
    COLUMNAS_MODELO = ("modelo", "número de parte", "numero de parte", "part number", "id")

    # tamaño máximo de las imágenes dentro del PDF (px a 96 dpi ≈ ancho de la columna)
    FOTO_MAX = (260, 230)
    VISTA_MAX = (260, 200)
    LOGO_ALTO = 40

    def __init__(self, rutas_config: dict | None = None, mapeo_categorias: dict | None = None,
                 crear_carpetas: bool = True, base_dir: str | None = None):
        """
        Inicializa el generador con un diccionario de rutas configuradas.
        crear_carpetas=False: no crea salidas_pdf/vistas_explosionadas (uso dentro del agente,
        donde el PDF se devuelve en memoria y no se guarda en disco).
        base_dir: carpeta contra la que se resuelven las rutas relativas (por defecto la
        carpeta de trabajo actual, como en la versión original).
        """
        self.config = rutas_config or {}
        mapeo = mapeo_categorias or self.MAPEO_CATEGORIAS_DEFAULT
        self.mapeo_categorias = {_clave(k): v for k, v in mapeo.items()}
        self.mapeo_categorias.setdefault("default", self.MAPEO_CATEGORIAS_DEFAULT["default"])

        def ruta(clave: str, defecto: str) -> str:
            p = self.config.get(clave, defecto)
            if base_dir and not os.path.isabs(p):
                p = os.path.join(base_dir, p)
            return os.path.abspath(p)

        self.carpeta_csv = ruta("datos_csv", "datos_csv")
        self.ruta_template = self._buscar_template(
            self.config.get("template_html", os.path.join("templates", "template.html")),
            ruta("template_html", os.path.join("templates", "template.html")))
        self.carpeta_fotos = ruta("fotos", "fotos")
        self.carpeta_logos = ruta("logos", "logos")
        self.carpeta_salidas = ruta("salidas_pdf", "salidas_pdf")
        self.carpeta_vistas = ruta("vistas_explosionadas", "vistas_explosionadas")

        if crear_carpetas:
            os.makedirs(self.carpeta_salidas, exist_ok=True)
            os.makedirs(self.carpeta_vistas, exist_ok=True)

    @staticmethod
    def _buscar_template(valor: str, principal: str) -> str:
        """Ruta de la plantilla HTML. Primero la resuelta contra base_dir (la carpeta del
        JSON); si no existe, la misma ruta relativa contra la carpeta del proyecto (padre de
        app/) y contra la carpeta de trabajo. Así "templates/template.html" en
        config/datasheet_pdf.json encuentra <proyecto>/templates/template.html.
        Si ninguna existe se avisa en el log y se usa PLANTILLA_DEFAULT."""
        if os.path.isfile(principal) or not valor or os.path.isabs(valor):
            if valor and not os.path.isfile(principal):
                log.warning("pdfdatagen: no existe la plantilla %s; se usa la integrada", principal)
            return principal
        raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for base in (raiz, os.getcwd()):
            cand = os.path.abspath(os.path.join(base, valor))
            if os.path.isfile(cand):
                return cand
        log.warning("pdfdatagen: no existe la plantilla %s (ni en %s); se usa la integrada",
                    principal, raiz)
        return principal

    @classmethod
    def desde_archivo_json(cls, ruta_json: str, **kwargs):
        """
        Método alternativo para instanciar la clase leyendo la configuración desde un archivo JSON.
        Las rutas relativas del JSON se resuelven contra la carpeta de trabajo actual
        (pasa base_dir=... para cambiarlo).
        """
        if not os.path.exists(ruta_json):
            raise FileNotFoundError(f"No se encontró el archivo de configuración JSON en: {ruta_json}")
        with open(ruta_json, "r", encoding="utf-8") as f:
            datos_config = json.load(f)
        return cls(rutas_config=datos_config.get("rutas", {}),
                   mapeo_categorias=datos_config.get("mapeo_categorias", None), **kwargs)

    # ------------------------------------------------------------------ búsqueda (standalone)
    def _buscar_producto_en_csvs(self, valor_buscado: str) -> dict | None:
        """Busca el producto en todos los CSV configurados. (Método Interno)"""
        import pandas as pd  # sólo lo necesita el modo standalone

        archivos_csv = sorted(
            f for ext in ("*.csv", "*.xlsx", "*.xlsm", "*.xls")
            for f in glob.glob(os.path.join(self.carpeta_csv, ext))
            if not os.path.basename(f).startswith("~$"))      # ~$x.xlsx: Excel abierto
        valor_buscado_str = str(valor_buscado).strip().lower()

        for ruta_csv in archivos_csv:
            try:
                if ruta_csv.lower().endswith((".xlsx", ".xlsm", ".xls")):
                    # todas las hojas, una debajo de otra (cada hoja con su encabezado)
                    hojas = pd.read_excel(ruta_csv, sheet_name=None, dtype=str)
                    df = pd.concat([h for h in hojas.values() if not h.empty],
                                   ignore_index=True) if hojas else pd.DataFrame()
                    if df.empty:
                        continue
                else:
                    try:
                        df = pd.read_csv(ruta_csv, sep=None, engine="python", encoding="utf-8-sig")
                    except UnicodeDecodeError:          # CSV exportado de Excel en Windows
                        df = pd.read_csv(ruta_csv, sep=None, engine="python", encoding="cp1252")
                df.columns = df.columns.astype(str)
                df.columns = df.columns.str.strip()
                columnas_candidatas = [c for c in df.columns if c.lower() in self.COLUMNAS_MODELO]
                if not columnas_candidatas:
                    continue

                coincidencias = pd.DataFrame()
                for col in columnas_candidatas:
                    df[col] = df[col].astype(str).str.strip()
                    filas_match = df[df[col].str.lower() == valor_buscado_str]
                    coincidencias = pd.concat([coincidencias, filas_match]).drop_duplicates()

                if not coincidencias.empty:
                    coincidencias = coincidencias.dropna(how="all", axis=1)
                    return {"data": coincidencias,
                            "categoria": os.path.splitext(os.path.basename(ruta_csv))[0].lower()}
            except Exception as e:
                log.warning("Error procesando %s: %s", ruta_csv, e)
        return None

    @staticmethod
    def _df_a_variantes(df) -> tuple[list[str], list[list[tuple[str, str]]]]:
        """DataFrame de variaciones -> (títulos de columna, filas (característica, valor))."""
        import pandas as pd
        titulos, variantes = [], []
        for i in range(len(df)):
            fila = df.iloc[i]
            titulos.append(str(fila.get("Número de parte", fila.get("Modelo", f"Variante {i + 1}"))))
            variantes.append([(str(c), "" if pd.isna(v) else str(v)) for c, v in fila.items()
                              if not str(c).startswith("Unnamed")])
        return titulos, variantes

    def _generar_tabla_html(self, df_variaciones) -> str:
        """Transpone el DataFrame de variaciones a código HTML estructurado. (Método Interno)"""
        titulos, variantes = self._df_a_variantes(df_variaciones)
        return self.tabla_html(variantes, titulos)

    # ------------------------------------------------------------------ HTML
    @staticmethod
    def tabla_html(variantes: Sequence[Variante], titulos: Sequence[str] | None = None) -> str:
        """Tabla 'Especificación | Modelo: A | Modelo: B ...' (una columna por variante).
        Los valores se escapan: un '<' o '&' del CSV no rompe el PDF."""
        e = html.escape
        orden: list[str] = []
        valores: list[dict[str, str]] = []
        for var in variantes:
            d = {}
            for c, v in var:
                c = str(c).strip()
                if not c or c.startswith("Unnamed"):
                    continue
                if c not in orden:
                    orden.append(c)
                d[c] = str(v).strip()
            valores.append(d)
        if not orden:
            return "<p>Sin especificaciones registradas.</p>"
        titulos = list(titulos or [f"Variante {i + 1}" for i in range(len(variantes))])
        out = ["<table class=\"specs\"><thead><tr><th>Especificación</th>"]
        out += [f"<th>Modelo: {e(t)}</th>" for t in titulos] if len(titulos) > 1 else ["<th>Valor</th>"]
        out.append("</tr></thead><tbody>")
        for i, c in enumerate(orden):
            out.append(f"<tr class=\"{'par' if i % 2 else 'non'}\"><td><strong>{e(c)}</strong></td>")
            for d in valores:
                out.append(f"<td>{e(d.get(c) or '-')}</td>")
            out.append("</tr>")
        out.append("</tbody></table>")
        return "".join(out)

    @staticmethod
    def _imagen(ruta: str | None, max_wh: tuple[int, int]) -> tuple[str, int, int]:
        """Imagen -> (data URI redimensionado, ancho, alto) que cabe en max_wh.
        Data URI en lugar de file://: funciona igual en Windows (rutas con espacios o
        letra de unidad) y en Linux. Sin imagen -> pixel blanco del tamaño de la caja."""
        if not ruta or not os.path.isfile(ruta):
            return _PIXEL_BLANCO, 1, 1
        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(ruta) as im:
                im.draft("RGB", (max_wh[0] * 3, max_wh[1] * 3))
                alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
                im = im.convert("RGBA" if alpha else "RGB")
                if alpha:                     # xhtml2pdf pinta mal la transparencia: fondo blanco
                    fondo = Image.new("RGB", im.size, "white")
                    fondo.paste(im, mask=im.split()[-1])
                    im = fondo
                im.thumbnail((max_wh[0] * 3, max_wh[1] * 3))     # 3x: nítida al imprimir
                w, h = im.size
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=88, optimize=True)
            esc = min(max_wh[0] / w, max_wh[1] / h)
            uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            return uri, max(1, int(w * esc)), max(1, int(h * esc))
        except Exception as ex:
            log.warning("pdfdatagen: no se pudo leer la imagen %s: %s", ruta, ex)
            return _PIXEL_BLANCO, 1, 1

    def _color(self, categoria: str) -> str:
        cfg = self.mapeo_categorias.get(_clave(categoria)) or self.mapeo_categorias["default"]
        return cfg.get("color", "#C62828")

    def _plantilla(self) -> str:
        if os.path.isfile(self.ruta_template):
            with open(self.ruta_template, "r", encoding="utf-8") as f:
                return f.read()
        return PLANTILLA_DEFAULT

    def construir_html(self, modelo: str, categoria: str, tabla_html: str,
                       ruta_foto: str | None, ruta_vista_explosionada: str | None,
                       ruta_logo: str | None, nombre: str = "") -> str:
        foto, fw, fh = self._imagen(ruta_foto, self.FOTO_MAX)
        vista, vw, vh = self._imagen(ruta_vista_explosionada, self.VISTA_MAX)
        logo, lw, lh = self._imagen(ruta_logo, (400, self.LOGO_ALTO))
        tiene_vista = vista != _PIXEL_BLANCO
        bloque_vista = (f'<div class="caja" style="margin-top: 4mm;"><img src="{vista}" '
                        f'width="{vw}" height="{vh}"></div>'
                        '<div class="cap">Vista explosionada</div>') if tiene_vista else ""
        e = html.escape
        return Template(self._plantilla()).safe_substitute(
            Modelo=e(modelo), Nombre=e(nombre or modelo), Categoria=e(categoria.capitalize()),
            Color_Acento=self._color(categoria), Fecha=f"{date.today():%d/%m/%Y}",
            Tabla_Especificaciones=tabla_html,
            Ruta_Logo=logo, Ruta_Foto=foto, Ruta_Vista_Explosionada=vista,
            Foto_W=fw, Foto_H=fh, Vista_W=vw, Vista_H=vh, Logo_H=lh,
            Bloque_Vista_Explosionada=bloque_vista)

    @staticmethod
    def html_a_pdf(html_final: str, destino) -> bool:
        """Compila el HTML a PDF (destino: archivo abierto en binario o BytesIO)."""
        from xhtml2pdf import pisa
        estado = pisa.CreatePDF(html_final, dest=destino, encoding="utf-8")
        return not estado.err

    # ------------------------------------------------------------------ API
    def generar_pdf_bytes(self, modelo: str, categoria: str, variantes: Sequence[Variante],
                          ruta_foto: str | None = None,
                          ruta_vista_explosionada: str | None = None,
                          ruta_logo: str | None = None, nombre: str = "",
                          titulos: Iterable[str] | None = None) -> bytes:
        """Compila el PDF en memoria con datos ya localizados (sin buscar en CSV).
        Lanza RuntimeError si xhtml2pdf no puede generarlo."""
        tabla = self.tabla_html(variantes, list(titulos) if titulos else [modelo])
        html_final = self.construir_html(modelo, categoria, tabla, ruta_foto,
                                         ruta_vista_explosionada, ruta_logo, nombre)
        buf = io.BytesIO()
        if not self.html_a_pdf(html_final, buf):
            raise RuntimeError(f"xhtml2pdf no pudo generar la ficha técnica de {modelo}")
        return buf.getvalue()

    def generar_pdf(self, modelo_a_buscar: str) -> str | None:
        """
        Ejecuta la búsqueda y exporta el archivo PDF final. Retorna la ruta del archivo generado.
        """
        resultado_busqueda = self._buscar_producto_en_csvs(modelo_a_buscar)
        if not resultado_busqueda:
            log.error("No se encontró '%s' en ningún archivo CSV.", modelo_a_buscar)
            return None

        df_producto = resultado_busqueda["data"]
        categoria = resultado_busqueda["categoria"]
        modelo_real = str(df_producto.iloc[0].get("Número de parte", modelo_a_buscar)).strip()

        def primera(*rutas: str) -> str | None:
            return next((r for r in rutas if os.path.isfile(r)), None)

        ruta_logo = primera(os.path.join(self.carpeta_logos, "logo-img.jpg"),
                            os.path.join(self.carpeta_logos, "default.jpg"))
        ruta_foto = primera(os.path.join(self.carpeta_fotos, f"{modelo_real}.jpg"),
                            os.path.join(self.carpeta_fotos, "default.jpg"))
        ruta_vista = primera(os.path.join(self.carpeta_vistas, f"{modelo_real}_explosionado.jpg"),
                             os.path.join(self.carpeta_vistas, "default.jpg"))

        html_final = self.construir_html(modelo_real, categoria,
                                         self._generar_tabla_html(df_producto),
                                         ruta_foto, ruta_vista, ruta_logo)

        os.makedirs(self.carpeta_salidas, exist_ok=True)
        nombre_archivo = f"Ficha_Tecnica_{categoria.upper()}_{modelo_real}.pdf"
        nombre_pdf_limpio = "".join("-" if ch in '/\\:*?"<>|' else ch for ch in nombre_archivo)
        ruta_salida_pdf = os.path.join(self.carpeta_salidas, nombre_pdf_limpio)

        with open(ruta_salida_pdf, "w+b") as result_file:
            ok = self.html_a_pdf(html_final, result_file)
        if ok:
            log.info("Ficha técnica creada: %s", ruta_salida_pdf)
            return ruta_salida_pdf
        log.error("Error al generar el PDF con xhtml2pdf (%s)", modelo_real)
        return None


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    cfg = sys.argv[2] if len(sys.argv) > 2 else "config.json"
    print(DatasheetGenerator.desde_archivo_json(cfg).generar_pdf(sys.argv[1]))
