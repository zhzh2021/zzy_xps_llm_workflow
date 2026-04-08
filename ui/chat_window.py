import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
from shiboken6 import isValid


from PySide6.QtCore import Qt, QThread, Signal, QObject, QEvent
from PySide6.QtGui import QKeyEvent, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QLabel, QComboBox, QSpinBox, QTextBrowser, QPlainTextEdit, QFrame, QSizePolicy,
    QMessageBox
)
from PySide6.QtGui import QScreen
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtCore import QMarginsF
from PySide6.QtGui import QPageLayout, QPageSize

from zzy_llm.ui.workers import ChatWorker, ExperimentExecWorker, ScriptExecWorker
from zzy_llm.llm_manager.ollama_utils import list_models, reset_session

logger = logging.getLogger(__name__)


# ---------------- UI helpers ----------------

def _bubble_html(text: str, sender: str) -> str:
    def esc(t: str) -> str:
        return (t.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace("\r\n", "\n").replace("\r", "\n")
                 .replace("\n", "<br>"))
    t = esc(text)

    if sender == "user":
        who, td_align, bg = "You", "right", "#DCF8C6"
    elif sender == "error":
        who, td_align, bg = "Error", "left", "#FFE5E5"
    else:
        who, td_align, bg = "Assistant", "left", "#FFFFFF"

    return f"""
<table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse; margin:0; padding:0;">
  <tr>
    <td style="text-align:{td_align}; vertical-align:top; padding:8px 0;">
      <div style="display:inline-block; max-width:74%; background:{bg};
                  border:1px solid #e5e5e5; border-radius:12px; padding:10px 12px;">
        <div style="color:#666; font-size:12px; margin:0 0 4px 0;"><b>{who}</b></div>
        <div style="font-size:14px; color:#111; line-height:1.35; margin:0;">{t}</div>
      </div>
    </td>
  </tr>
</table>
"""


def _read_text_file(path: Path, limit_bytes: int = 200_000) -> str:
    try:
        data = path.read_bytes()[:limit_bytes]
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"[Failed to read file: {e}]"


# ---------------- Main Window ----------------

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zzy LLM – Chat")
        self.setMinimumSize(580, 420)

        # State
        self.session_id = "main"
        self.attached_files: List[Path] = []
        self.in_experiment = False
        self._message_log: List[tuple] = []  # (sender, text) for PDF export

        # Top bar
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)

        self.modelLabel = QLabel("Model:")
        self.modelCombo = QComboBox()
        try:
            avail_models = list_models()
        except Exception:
            avail_models = []
        if not avail_models:
            # keep UI usable if ollama CLI is not available
            avail_models = ["No models found"]
        self.modelCombo.addItems(avail_models)

        self.tokensLabel = QLabel("Max tokens:")
        self.tokensSpin = QSpinBox()
        self.tokensSpin.setRange(32, 16000)
        self.tokensSpin.setValue(5000)

        self.attachBtn = QPushButton("Attach files…")
        self.attachBtn.clicked.connect(self.choose_files)
        self.clearBtn = QPushButton("Clear")
        self.clearBtn.clicked.connect(self.clear_chat)
        self.screenshotBtn = QPushButton("📷 Export")
        self.screenshotBtn.setToolTip("Save a high-resolution screenshot (publication quality)")
        self.screenshotBtn.clicked.connect(self.export_screenshot)
        self.pdfBtn = QPushButton("📄 PDF")
        self.pdfBtn.setToolTip("Export full conversation as PDF (captures all scrolled content)")
        self.pdfBtn.clicked.connect(self.export_pdf)

        top.addWidget(self.modelLabel)
        top.addWidget(self.modelCombo)
        top.addSpacing(8)
        top.addWidget(self.tokensLabel)
        top.addWidget(self.tokensSpin)
        top.addStretch(1)
        top.addWidget(self.attachBtn)
        top.addWidget(self.clearBtn)
        top.addWidget(self.screenshotBtn)
        top.addWidget(self.pdfBtn)

        # Attached files line
        self.attachLine = QLabel("")
        self.attachLine.setWordWrap(True)
        self.attachLine.setStyleSheet("color:#444; font-size:12px;")
        self.attachLine.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Divider
        div1 = QFrame()
        div1.setFrameShape(QFrame.HLine)
        div1.setFrameShadow(QFrame.Sunken)

        # History
        self.history = QTextBrowser()
        self.history.setOpenExternalLinks(True)
        self.history.setStyleSheet(
            "QTextBrowser { background:#F7F7F8; border:1px solid #ddd; }")
        self.history.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Input row
        inputRow = QHBoxLayout()
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText(
            "Type a message… (Shift+Enter for newline, Enter to send)")
        self.input.setMaximumBlockCount(2000)
        self.input.installEventFilter(self)

        self.sendBtn = QPushButton("Send")
        self.sendBtn.clicked.connect(self.on_send_clicked)

        inputRow.addWidget(self.input, 1)
        inputRow.addWidget(self.sendBtn, 0)

        # Layout
        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(self.attachLine)
        root.addWidget(div1)
        root.addWidget(self.history, 1)
        root.addLayout(inputRow)

        # threading
        self.thread: Optional[QThread] = None
        self.worker: Optional[QObject] = None

        # greeting
        self.append_assistant("Hi! I am your XPS data assistant. I am connected to your local Ollama via ChatOllama. "
                              "Attach .txt/.md files or just type and press Enter.")

    # -------- Screenshot export --------

    def export_screenshot(self):
        """Capture the window at full device pixel ratio and save as PNG."""
        default_name = "xps_agent_demo.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", default_name, "PNG Image (*.png)"
        )
        if not path:
            return
        screen: QScreen = self.screen()
        # grabWindow at native device pixel ratio → crisp on HiDPI / retina
        pixmap = screen.grabWindow(int(self.winId()))
        if pixmap.save(path, "PNG"):
            QMessageBox.information(
                self, "Screenshot saved",
                f"Saved {pixmap.width()}×{pixmap.height()} px image to:\n{path}"
            )
        else:
            QMessageBox.warning(self, "Save failed", f"Could not write to:\n{path}")

    def export_pdf(self):
        """Render the full conversation to a PDF using Qt-native HTML (crisp, readable)."""
        import html as _html
        default_name = "xps_agent_conversation.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF", default_name, "PDF Document (*.pdf)"
        )
        if not path:
            return

        # Build Qt-compatible HTML (no CSS3, no inline-block) from message log
        def _print_bubble(sender: str, text: str) -> str:
            t = _html.escape(text).replace("\n", "<br>")
            if sender == "user":
                bg, label, align = "#d4f0da", "You", "right"
            elif sender == "error":
                bg, label, align = "#ffe5e5", "Error", "left"
            else:
                bg, label, align = "#f2f2f2", "Assistant", "left"
            return (
                f'<table width="100%" cellspacing="0" cellpadding="0">'
                f'<tr><td align="{align}">'
                f'<table cellspacing="0" cellpadding="10" width="85%"'
                f' bgcolor="{bg}" style="border:1px solid #bbb; margin:8px 0;">'
                f'<tr><td><font size="3" color="#555"><b>{label}</b></font></td></tr>'
                f'<tr><td><font size="4" color="#111">{t}</font></td></tr>'
                f'</table></td></tr></table>'
            )

        parts = [_print_bubble(s, t) for s, t in self._message_log]
        body = "<html><body>" + "".join(parts) + "</body></html>"

        from PySide6.QtWidgets import QTextBrowser as _TB
        renderer = _TB()
        renderer.resize(900, 2000)
        renderer.setHtml(body)

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(path)
        printer.setPageLayout(QPageLayout(
            QPageSize(QPageSize.PageSizeId.A4),
            QPageLayout.Orientation.Portrait,
            QMarginsF(20, 20, 20, 20),
            QPageLayout.Unit.Millimeter,
        ))
        renderer.print_(printer)

        QMessageBox.information(
            self, "PDF saved",
            f"Full conversation exported to:\n{path}"
        )

    # -------- Chat actions --------

    def on_send_clicked(self):
        if not self.sendBtn.isEnabled() or self.in_experiment:
            return
        text = self.input.toPlainText().strip()
        if not text:
            return

        # inline attachments
        attachments_text = ""
        for p in self.attached_files:
            suffix = p.suffix.lower()
            if suffix in (".txt", ".md", ".log", ".csv"):
                content = _read_text_file(p)
                attachments_text += f"\n\n[File: {p.name}]\n{content}\n"
            else:
                attachments_text += f"\n\n[File: {p.name}] (unsupported type for inline reading)\n"

        full_prompt = text + \
            (f"\n\n### Attached Context:{attachments_text}" if attachments_text else "")

        # show user bubble
        self.append_user(text)
        if attachments_text:
            self.append_user(
                f"(Attached {len(self.attached_files)} file(s) included in prompt)")

        # disable UI during request
        self.set_ui_enabled(False)

        # spin worker (router-only; model comes from combo)
        model_name = self.modelCombo.currentText()
        max_tokens = self.tokensSpin.value()

        self.thread = QThread()
        self.worker = ChatWorker(
            router_model_name=model_name, prompt=full_prompt, max_tokens=max_tokens)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)

        # signals
        self.worker.propose.connect(self.on_proposal_experiment)
        self.worker.propose_execute.connect(self.on_proposal_execute)
        if hasattr(self.worker, "triage_summary"):
            self.worker.triage_summary.connect(self.on_triage_summary)
        if hasattr(self.worker, "status"):
            self.worker.status.connect(self.on_status_message)
        self.worker.finished.connect(self.on_response_ready)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

        self.input.clear()

    def on_response_ready(self, response: str, error: str):
        if error:
            self.append_error(error)
        elif response:
            self.append_assistant(response)
        if not self.in_experiment:
            self.set_ui_enabled(True)

    def on_proposal_experiment(self, proposal: dict):
        # proposal: {"experiment": str, "args": dict, "message": str}
        exp = proposal.get("experiment", "")
        args = proposal.get("args") or {}
        msg = proposal.get("message", "")

        pretty_args = ", ".join(
            f"{k}={repr(v)}" for k, v in args.items()) or "(no args)"
        text = f"Run {exp} with args {pretty_args}?"
        if msg:
            text += f"\n{msg}"

        reply = QMessageBox.question(
            self, "Confirm experiment", text,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            self.in_experiment = True
            self.append_assistant(f"Running **{exp}**…")
            self._run_experiment_async(exp, args)
        else:
            self.append_assistant("Cancelled.")
            self.set_ui_enabled(True)

        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

    def on_proposal_execute(self, payload: dict):
        # payload: {"message": str, "script": {"language":..., "code":...}, "ui_confirm": {...}}
        message = payload.get("message", "")
        script = payload.get("script", {}) or {}
        ui = payload.get("ui_confirm", {}) or {}

        lang = script.get("language") or "python"
        code = script.get("code") or ""

        title = ui.get("title") or "Confirm script execution"
        summary = ui.get("summary") or message
        risk = ui.get("risk_level") or "low"
        requires_admin = ui.get("requires_admin", False)
        effects = ui.get("estimated_effects", [])

        details = f"{summary}\n\nLanguage: {lang}\nRisk: {risk}{' (admin)' if requires_admin else ''}"
        if effects:
            details += "\nEffects: " + ", ".join(effects)
        details += "\n\n--- Script ---\n" + code

        reply = QMessageBox.question(
            self, title, details,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.in_experiment = True
            self.append_assistant("Executing script…")
            self._run_script_async(lang, code)
        else:
            self.append_assistant("Script cancelled.")
            self.set_ui_enabled(True)

        if self.thread is not None and isValid(self.thread):
            if self.thread.isRunning():
                print("Thread still running")
                return


    # -------- Attachments --------

    def choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Attach files", str(Path.home()),
            "Text files (*.txt *.md *.log *.csv);;All files (*.*)"
        )
        if not files:
            return
        existing = set(map(str, self.attached_files))
        for f in files:
            if f not in existing:
                self.attached_files.append(Path(f))
        self.refresh_attachment_line()

    def refresh_attachment_line(self):
        if not self.attached_files:
            self.attachLine.setText("")
            return
        names = "  •  ".join(p.name for p in self.attached_files[:8])
        more = len(self.attached_files) - 8
        if more > 0:
            names += f" (+{more} more)"
        self.attachLine.setText(f"Attached: {names}")

    # -------- Chat history rendering --------

    def _insert_bubble(self, html: str):
        c = self.history.textCursor()
        c.movePosition(QTextCursor.End)
        self.history.setTextCursor(c)
        self.history.textCursor().insertBlock()
        self.history.insertHtml(html)
        self.history.textCursor().insertBlock()
        self.history.ensureCursorVisible()

    def append_user(self, text: str):
        self._message_log.append(("user", text))
        self._insert_bubble(_bubble_html(text, "user"))

    def append_assistant(self, text: str):
        self._message_log.append(("assistant", text))
        self._insert_bubble(_bubble_html(text, "assistant"))

    def append_error(self, text: str):
        self._message_log.append(("error", text))
        self._insert_bubble(_bubble_html(text, "error"))

    def on_triage_summary(self, summary: str):
        if summary:
            self.append_assistant(summary)

    def on_status_message(self, message: str):
        if message:
            self.append_assistant(message)

    # -------- Misc --------

    def clear_chat(self):
        self._message_log.clear()
        self.history.clear()
        self.attached_files.clear()
        self.refresh_attachment_line()
        current_model = self.modelCombo.currentText()
        if current_model:
            reset_session(self.session_id, current_model)
        self.append_assistant("Cleared. Start a new chat when ready.")

    def set_ui_enabled(self, enabled: bool):
        self.sendBtn.setEnabled(enabled)
        self.attachBtn.setEnabled(enabled)
        self.modelCombo.setEnabled(enabled)
        self.tokensSpin.setEnabled(enabled)
        self.input.setReadOnly(not enabled)

    # Enter to send, Shift+Enter for newline
    def eventFilter(self, obj, event):
        if obj is self.input and event.type() == QEvent.KeyPress:
            e: QKeyEvent = event  # type: ignore
            if (e.key() in (Qt.Key_Return, Qt.Key_Enter)) and not (e.modifiers() & Qt.ShiftModifier):
                self.on_send_clicked()
                return True
        return super().eventFilter(obj, event)

    # Async runners
    def _run_experiment_async(self, exp: str, args: dict):
        self.execThread = QThread()
        self.execWorker = ExperimentExecWorker(exp, args)
        self.execWorker.moveToThread(self.execThread)
        self.execThread.started.connect(self.execWorker.run)
        self.execWorker.done.connect(self.on_experiment_done)
        self.execWorker.done.connect(self.execThread.quit)
        self.execWorker.done.connect(self.execWorker.deleteLater)
        self.execThread.finished.connect(self.execThread.deleteLater)
        self.execThread.start()

    def on_experiment_done(self, message, error):
        if error:
            self.append_error(error)
        else:
            self.append_assistant(message)
        self.in_experiment = False
        self.set_ui_enabled(True)

    def _run_script_async(self, language: str, code: str):
        self.execThread = QThread()
        self.execWorker = ScriptExecWorker(language, code)
        self.execWorker.moveToThread(self.execThread)
        self.execThread.started.connect(self.execWorker.run)
        self.execWorker.done.connect(self.on_script_done)
        self.execWorker.done.connect(self.execThread.quit)
        self.execWorker.done.connect(self.execWorker.deleteLater)
        self.execThread.finished.connect(self.execThread.deleteLater)
        self.execThread.start()

    def on_script_done(self, message, error):
        if error:
            self.append_error(error)
        else:
            self.append_assistant(message)
        self.in_experiment = False
        self.set_ui_enabled(True)


# ---------- Demo conversations (--demo / --demo-quality flags) ----------

_DEMO_CONVERSATION = [
    ("user",
     "I have a new batch of Si 2p spectra collected after HF etching. "
     "The raw .spe files are in project_root/00_raw_data/Si2p_HF. "
     "Can you convert, fit, and quantify them, then generate summary plots?"),
    ("assistant",
     "Sure! I've detected 12 .spe files in Si2p_HF. Here is the plan I will execute:\n"
     "① Convert all .spe → CSV (XPS_reader)\n"
     "② Apply Shirley background + fit Si⁰/Si⁴⁺ components using the Si2p template (XPS_Fitter)\n"
     "③ Quantify atomic % from peak areas (XPS_Quantifier)\n"
     "④ Export overlay and individual fit plots to 04_plots/Si2p_HF (XPS_Plotter)\n\n"
     "Proceeding now — I'll report back when each step completes."),
    ("user",
     "Also correlate the Si⁰/Si⁴⁺ ratio with the etching time variable in the sample sheet."),
    ("assistant",
     "Correlation analysis added to the queue. I'll load the etching-time metadata from "
     "xps_config/sample_sheet.csv and compute the Pearson r between Si⁰/(Si⁰+Si⁴⁺) ratio "
     "and etch duration. Results will be saved to 06_correlator_results/Si2p_etch_correlation.csv "
     "with a scatter plot + regression line."),
]

_DEMO_QUALITY_CONVERSATION = [
    ("user",
     "Please run the full XPS workflow on the Fe 2p batch in "
     "project_root/00_raw_data/Fe2p_batch03 — convert, quality check, fit, quantify."),
    ("assistant",
     "Starting workflow for Fe 2p batch03. Running Quality Gate (Level 1 + Level 2) "
     "on all 9 spectra before fitting…\n\n"
     "✅ Fe2p_sample01.xy  —  GOOD   (SNR: 48.3, HR scan, suitable for fitting)\n"
     "✅ Fe2p_sample02.xy  —  GOOD   (SNR: 51.7, HR scan, suitable for fitting)\n"
     "✅ Fe2p_sample03.xy  —  EXCELLENT  (SNR: 67.1)\n"
     "✅ Fe2p_sample04.xy  —  ACCEPTABLE  (SNR: 22.4, 1 warning)\n"
     "⚠️ Fe2p_sample05.xy  —  POOR   (SNR: 8.1 — below threshold 10.0)\n"
     "✅ Fe2p_sample06.xy  —  GOOD   (SNR: 44.9)\n"
     "⚠️ Fe2p_sample07.xy  —  POOR   (SNR: 6.3 — below threshold 10.0)\n"
     "✅ Fe2p_sample08.xy  —  GOOD   (SNR: 39.2)\n"
     "🚨 Fe2p_sample09.xy  —  CRITICAL  (energy axis gap detected — possible beam instability)"),
    ("assistant",
     "⚠️ Quality Gate raised 3 flags before fitting was attempted:\n\n"
     "[WARNING] Fe2p_sample05.xy — SNR: 8.1\n"
     "  Category: SNR  |  value=8.1, threshold=10.0\n"
     "  Fitting will be unreliable; peak positions may shift by >0.3 eV.\n\n"
     "[WARNING] Fe2p_sample07.xy — SNR: 6.3\n"
     "  Category: SNR  |  value=6.3, threshold=10.0\n"
     "  Recommend re-collecting with longer dwell time (≥500 ms/step).\n\n"
     "[CRITICAL] Fe2p_sample09.xy — Energy axis gap detected\n"
     "  Category: Energy  |  Gap of 2.8 eV found between points 312–313\n"
     "  Possible beam instability or scan abort mid-acquisition.\n"
     "  This file has been excluded from fitting automatically.\n\n"
     "📄 Full quality report saved to:\n"
     "  project_root/_logs/quality_report_Fe2p_batch03_20260407.json\n\n"
     "Please open the report to scrutinize the 3 flagged spectra before "
     "accepting the batch results. I can display the problem spectra "
     "side-by-side if you'd like."),
    ("user",
     "Show me sample05 and sample09 overlaid so I can decide."),
    ("assistant",
     "Generating overlay plot of Fe2p_sample05 (POOR SNR) and Fe2p_sample09 (CRITICAL — energy gap) "
     "against Fe2p_sample03 (EXCELLENT) as reference…\n\n"
     "Plot saved → 04_plots/Fe2p_batch03/quality_flag_overlay.png\n\n"
     "Observations from the overlay:\n"
     "• sample05: counts ~3× lower than reference; Fe²⁺/Fe³⁺ shoulder barely resolved — "
     "re-collect recommended before publication.\n"
     "• sample09: sharp discontinuity at 710.8 eV confirms scan abort. "
     "Exclude from quantification or re-acquire.\n\n"
     "The remaining 6 spectra (sample01–04, 06, 08) passed the quality gate. "
     "Shall I proceed with fitting those now?"),
]


def _load_demo(w: "ChatWindow", conversation: list):
    """Populate chat history with a pre-built demo conversation."""
    w.history.clear()
    w._message_log.clear()
    for sender, text in conversation:
        if sender == "user":
            w.append_user(text)
        else:
            w.append_assistant(text)


# ---------- App entry ----------
def main(demo: bool = False, demo_quality: bool = False):
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_SCALE_FACTOR", "1")  # set to "2" for 2× crisp rendering
    app = QApplication(sys.argv)
    w = ChatWindow()
    if demo_quality or "--demo-quality" in sys.argv:
        _load_demo(w, _DEMO_QUALITY_CONVERSATION)
    elif demo or "--demo" in sys.argv:
        _load_demo(w, _DEMO_CONVERSATION)
    w.resize(700, 520)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
