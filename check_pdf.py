"""Headless script to generate a print-optimized PDF for inspection."""
import sys, os, html as _html

os.environ["QT_SCALE_FACTOR"] = "1"
from PySide6.QtWidgets import QApplication, QTextBrowser
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtGui import QPageLayout, QPageSize
from PySide6.QtCore import QMarginsF

app = QApplication(sys.argv)

from zzy_llm.ui.chat_window import _DEMO_QUALITY_CONVERSATION


def bubble(sender: str, text: str) -> str:
    """Qt-HTML-compatible bubble — no inline-block / border-radius."""
    t = _html.escape(text).replace("\n", "<br>")
    if sender == "user":
        bg, label, align = "#d4f0da", "You", "right"
    else:
        bg, label, align = "#f2f2f2", "Assistant", "left"
    return (
        f'<table width="100%" cellspacing="0" cellpadding="0">'
        f'<tr><td align="{align}">'
        f'<table cellspacing="0" cellpadding="10" width="82%"'
        f' bgcolor="{bg}" style="border:1px solid #bbb; margin:8px 0;">'
        f'<tr><td><font size="3" color="#555"><b>{label}</b></font></td></tr>'
        f'<tr><td><font size="4" color="#111">{t}</font></td></tr>'
        f'</table></td></tr></table>'
    )


parts = [bubble(s, t) for s, t in _DEMO_QUALITY_CONVERSATION]
body = "<html><body>" + "".join(parts) + "</body></html>"

browser = QTextBrowser()
browser.resize(900, 2000)
browser.setHtml(body)

out = r"C:\Users\b82797\Documents\Github\zz_llm\demo_quality_check_v2.pdf"
printer = QPrinter(QPrinter.PrinterMode.HighResolution)
printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
printer.setOutputFileName(out)
printer.setPageLayout(QPageLayout(
    QPageSize(QPageSize.PageSizeId.A4),
    QPageLayout.Orientation.Portrait,
    QMarginsF(20, 20, 20, 20),
    QPageLayout.Unit.Millimeter,
))
browser.print_(printer)
print(f"PDF written: {out}  ({os.path.getsize(out):,} bytes)")
app.quit()
