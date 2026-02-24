from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT_DIR = Path("data/pdfs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

docs = [
    ("politica_vacaciones.pdf", [
        "Empresa Ficticia: NovaWorks S.L.",
        "POLITICA DE VACACIONES (Version 1.0)",
        "",
        "1. Derecho a vacaciones:",
        "- 23 dias laborables por ano para empleados a tiempo completo.",
        "- La antiguedad no incrementa dias salvo convenio especifico.",
        "",
        "2. Solicitud y aprobacion:",
        "- Solicitar con al menos 15 dias de antelacion en el portal interno.",
        "- El manager aprueba o propone alternativa en un maximo de 5 dias.",
        "",
        "3. Periodos restringidos:",
        "- En cierre fiscal (1-15 enero) se limitan ausencias en Finanzas.",
        "",
        "4. Vacaciones no disfrutadas:",
        "- Deben disfrutarse antes del 31 de marzo del ano siguiente.",
    ]),
    ("manual_prl.pdf", [
        "Empresa Ficticia: NovaWorks S.L.",
        "MANUAL DE PREVENCION DE RIESGOS LABORALES (PRL) (Version 1.0)",
        "",
        "1. Equipos de proteccion individual (EPI):",
        "- En almacen es obligatorio casco, chaleco y calzado de seguridad.",
        "",
        "2. Reporte de incidentes:",
        "- Cualquier incidente o cuasi-incidente debe reportarse en 24 horas.",
        "",
        "3. Ergonomia en oficina:",
        "- Ajustar silla para que caderas y rodillas formen angulo de 90 grados.",
        "- Pausas de 5 minutos cada 60 minutos de trabajo con pantalla.",
        "",
        "4. Evacuacion:",
        "- Seguir rutas de evacuacion señalizadas; punto de reunion: parking norte.",
    ]),
    ("protocolo_seguridad_it.pdf", [
        "Empresa Ficticia: NovaWorks S.L.",
        "PROTOCOLO DE SEGURIDAD INFORMATICA (Version 1.0)",
        "",
        "1. Contrasenas:",
        "- Longitud minima 12 caracteres, con mayusculas, minusculas y numeros.",
        "- Rotacion cada 90 dias para cuentas privilegiadas.",
        "",
        "2. Acceso remoto:",
        "- VPN obligatoria y MFA (doble factor) habilitado.",
        "",
        "3. Datos y clasificacion:",
        "- 'Confidencial' no puede enviarse por email sin cifrado.",
        "",
        "4. Incidentes de seguridad:",
        "- Notificar al SOC en menos de 1 hora tras deteccion.",
    ]),
    ("codigo_conducta.pdf", [
        "Empresa Ficticia: NovaWorks S.L.",
        "CODIGO DE CONDUCTA (Version 1.0)",
        "",
        "1. Comportamiento profesional:",
        "- Respeto, inclusion y colaboracion son obligatorios.",
        "",
        "2. Conflictos de interes:",
        "- Declarar cualquier relacion comercial con proveedores.",
        "",
        "3. Canales de denuncia:",
        "- Canal confidencial disponible 24/7 (compliance@novaworks.example).",
    ]),
]

def write_pdf(path: Path, lines: list[str]):
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 60
    for line in lines:
        c.drawString(60, y, line)
        y -= 16
        if y < 60:
            c.showPage()
            y = height - 60
    c.save()

def main():
    for filename, lines in docs:
        write_pdf(OUT_DIR / filename, lines)
    print(f"OK: PDFs generados en {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
