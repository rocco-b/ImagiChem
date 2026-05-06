import csv
import io
import os
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from rdkit import Chem
from rdkit.Chem import Draw

import imagichem_core


RESULTS_PER_PAGE = 48
APP_NAME = "ImagiChem"


# -----------------------------------------------------------------------------
# Resource and image helpers
# -----------------------------------------------------------------------------
def resource_path(relative_path: str) -> str:
    """Return a resource path that works in development and PyInstaller builds."""
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def pixmap_from_path(path: str) -> QPixmap:
    """Load a QPixmap from paths containing spaces, accents, or OneDrive Unicode names."""
    pixmap = QPixmap(path)
    if not pixmap.isNull():
        return pixmap
    try:
        data = Path(path).read_bytes()
        pixmap = QPixmap()
        pixmap.loadFromData(data)
        return pixmap
    except Exception:
        return QPixmap()


# -----------------------------------------------------------------------------
# Modern application style
# -----------------------------------------------------------------------------
STYLESHEET = """
QWidget {
    background-color: #F6F8FB;
    color: #1E293B;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12pt;
}
QMainWindow {
    background-color: #F6F8FB;
}
QLabel#TitleLabel {
    font-size: 30pt;
    font-weight: 800;
    color: #2636D9;
}
QLabel#SubtitleLabel {
    font-size: 11pt;
    color: #64748B;
}
QLabel#SectionLabel {
    font-size: 13pt;
    font-weight: 700;
    color: #0F172A;
}
QLabel#ImagePreviewLabel {
    border: 2px dashed #CBD5E1;
    border-radius: 18px;
    background-color: #FFFFFF;
    color: #64748B;
}
QLabel#SmallInfoLabel {
    color: #475569;
    font-size: 10pt;
}
QFrame#PanelFrame {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
}
QFrame#CardFrame {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
}
QPushButton {
    background-color: #2636D9;
    color: white;
    border: none;
    padding: 10px 18px;
    border-radius: 10px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #3F4EF0;
}
QPushButton:pressed {
    background-color: #1D2BB5;
}
QPushButton:disabled {
    background-color: #CBD5E1;
    color: #64748B;
}
QPushButton#SecondaryButton {
    background-color: #E2E8F0;
    color: #0F172A;
}
QPushButton#SecondaryButton:hover {
    background-color: #CBD5E1;
}
QPushButton#DangerButton {
    background-color: #EF4444;
    color: white;
}
QComboBox, QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    padding: 8px;
    min-height: 24px;
}
QTextEdit {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 6px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9pt;
}
QProgressBar {
    border: 1px solid #CBD5E1;
    border-radius: 9px;
    background-color: #FFFFFF;
    text-align: center;
    font-weight: 600;
    min-height: 20px;
}
QProgressBar::chunk {
    border-radius: 8px;
    background-color: #2636D9;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QStatusBar {
    background-color: #FFFFFF;
    border-top: 1px solid #E2E8F0;
}
"""


# -----------------------------------------------------------------------------
# Background worker
# -----------------------------------------------------------------------------
class ImagichemWorker(QObject):
    finished = pyqtSignal(list)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, image_path: str, generation_mode: str = "hybrid"):
        super().__init__()
        self.image_path = image_path
        self.generation_mode = generation_mode

    def run(self):
        try:
            results = imagichem_core.run_imagichem_processing(
                self.image_path,
                self.progress.emit,
                self.generation_mode,
            )
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(f"An unexpected error occurred: {exc}")


# -----------------------------------------------------------------------------
# Result card widgets
# -----------------------------------------------------------------------------
class ScoreBar(QWidget):
    def __init__(self, score: float = 0.0):
        super().__init__()
        self.score = max(0.0, min(float(score), 100.0))
        self.setMinimumHeight(22)

    def set_score(self, score: float):
        self.score = max(0.0, min(float(score), 100.0))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setBrush(QColor("#E2E8F0"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 9, 9)

        if self.score >= 75:
            color = QColor("#16A34A")
        elif self.score >= 60:
            color = QColor("#84CC16")
        elif self.score >= 45:
            color = QColor("#F59E0B")
        else:
            color = QColor("#EF4444")

        width = int(rect.width() * (self.score / 100.0))
        if width > 0:
            painter.setBrush(color)
            painter.drawRoundedRect(rect.x(), rect.y(), width, rect.height(), 9, 9)

        painter.setPen(QColor("#0F172A"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{self.score:.2f}")


class ResultCard(QFrame):
    def __init__(self, smiles: str, score: float, image_pixmap: QPixmap):
        super().__init__()
        self.setObjectName("CardFrame")
        self.smiles_string = smiles

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.image_label = QLabel()
        self.image_label.setPixmap(image_pixmap)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(210, 210)

        score_label = QLabel("Drug-likeness score")
        score_label.setObjectName("SmallInfoLabel")
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.score_bar = ScoreBar(score)

        self.smiles_preview = QTextEdit()
        self.smiles_preview.setPlainText(smiles)
        self.smiles_preview.setReadOnly(True)
        self.smiles_preview.setFixedHeight(58)

        copy_button = QPushButton("Copy SMILES")
        copy_button.setObjectName("SecondaryButton")
        copy_button.clicked.connect(lambda: self.copy_smiles_to_clipboard(copy_button))

        layout.addWidget(self.image_label)
        layout.addWidget(score_label)
        layout.addWidget(self.score_bar)
        layout.addWidget(self.smiles_preview)
        layout.addWidget(copy_button)

    def copy_smiles_to_clipboard(self, button: QPushButton):
        QApplication.clipboard().setText(self.smiles_string)
        old_text = button.text()
        button.setText("Copied")
        QTimer.singleShot(1500, lambda: button.setText(old_text))


# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImagiChem - Image-Conditioned Molecule Generation")
        self.setMinimumSize(1220, 820)
        self.resize(1320, 900)
        self.setStyleSheet(STYLESHEET)

        icon_path = resource_path("icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.image_path: str | None = None
        self.results: list[tuple[str, float]] = []
        self.filtered_results: list[tuple[str, float]] = []
        self.image_cache: dict[str, QPixmap] = {}
        self.current_page = 0
        self.thread: QThread | None = None
        self.worker: ImagichemWorker | None = None

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.init_home_page()
        self.init_results_page()
        self.init_footer()

        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.results_page)

    # ------------------------- Layout builders -------------------------
    def init_footer(self):
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self.footer_label = QLabel(
            "Designed and developed by Prof. Antonio Rescifina and Dr. Rocco Buccheri — University of Catania"
        )
        self.footer_label.setObjectName("SmallInfoLabel")
        status_bar.addWidget(self.footer_label, 1)

    def init_home_page(self):
        self.home_page = QWidget()
        page_layout = QVBoxLayout(self.home_page)
        page_layout.setContentsMargins(40, 28, 40, 28)
        page_layout.setSpacing(18)

        title = QLabel("ImagiChem")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("Deterministic image-conditioned generation of chemically valid and drug-like molecules")
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        center = QHBoxLayout()
        center.setSpacing(24)

        left_panel = QFrame()
        left_panel.setObjectName("PanelFrame")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(24, 24, 24, 24)
        left_layout.setSpacing(14)

        self.image_preview = QLabel("Drop-in preview\nSelect an image to start")
        self.image_preview.setObjectName("ImagePreviewLabel")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setFixedSize(430, 430)

        self.image_path_label = QLabel("No image selected")
        self.image_path_label.setObjectName("SmallInfoLabel")
        self.image_path_label.setWordWrap(True)
        self.image_path_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        image_buttons = QHBoxLayout()
        self.select_button = QPushButton("Open Image")
        self.select_button.clicked.connect(self.select_image)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("SecondaryButton")
        self.clear_button.clicked.connect(self.reset_home_state)
        image_buttons.addWidget(self.select_button)
        image_buttons.addWidget(self.clear_button)

        left_layout.addWidget(self.image_preview, alignment=Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.image_path_label)
        left_layout.addLayout(image_buttons)

        right_panel = QFrame()
        right_panel.setObjectName("PanelFrame")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(28, 28, 28, 28)
        right_layout.setSpacing(14)

        section = QLabel("Generation settings")
        section.setObjectName("SectionLabel")

        mode_help = QLabel(
            "Choose one of the three integrated backends. Hybrid combines the library-based engine with the image-conditioned from-scratch generator."
        )
        mode_help.setObjectName("SmallInfoLabel")
        mode_help.setWordWrap(True)

        self.mode_selector = QComboBox()
        self.mode_selector.addItem("Hybrid — recommended", "hybrid")
        self.mode_selector.addItem("Library only", "library")
        self.mode_selector.addItem("From-scratch only", "fromscratch")
        self.mode_selector.setCurrentIndex(0)

        self.start_button = QPushButton("Generate Molecules")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_analysis)

        self.open_results_button = QPushButton("Open Results")
        self.open_results_button.setObjectName("SecondaryButton")
        self.open_results_button.hide()
        self.open_results_button.clicked.connect(self.show_results_page)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()

        self.status_message = QLabel("Ready")
        self.status_message.setObjectName("SmallInfoLabel")
        self.status_message.setWordWrap(True)

        info = QLabel(
            "Output: ranked SMILES with molecular depictions. Results can be saved as .smi or .csv without closing the program."
        )
        info.setObjectName("SmallInfoLabel")
        info.setWordWrap(True)

        right_layout.addWidget(section)
        right_layout.addWidget(mode_help)
        right_layout.addWidget(self.mode_selector)
        right_layout.addSpacing(10)
        right_layout.addWidget(self.start_button)
        right_layout.addWidget(self.open_results_button)
        right_layout.addWidget(self.progress_bar)
        right_layout.addWidget(self.status_message)
        right_layout.addStretch(1)
        right_layout.addWidget(info)

        center.addWidget(left_panel, 0)
        center.addWidget(right_panel, 1)

        page_layout.addWidget(title)
        page_layout.addWidget(subtitle)
        page_layout.addSpacing(8)
        page_layout.addLayout(center)

    def init_results_page(self):
        self.results_page = QWidget()
        layout = QVBoxLayout(self.results_page)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(10)

        self.new_image_button = QPushButton("Open New Image")
        self.new_image_button.clicked.connect(self.open_new_image)

        self.back_button = QPushButton("Back to Setup")
        self.back_button.setObjectName("SecondaryButton")
        self.back_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.home_page))

        self.prev_button = QPushButton("Previous")
        self.prev_button.setObjectName("SecondaryButton")
        self.prev_button.clicked.connect(self.prev_page)

        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("SecondaryButton")
        self.next_button.clicked.connect(self.next_page)

        self.save_smi_button = QPushButton("Save SMILES")
        self.save_smi_button.clicked.connect(self.save_smiles)

        self.save_csv_button = QPushButton("Save CSV")
        self.save_csv_button.setObjectName("SecondaryButton")
        self.save_csv_button.clicked.connect(self.save_csv)

        top_bar.addWidget(self.new_image_button)
        top_bar.addWidget(self.back_button)
        top_bar.addStretch(1)
        top_bar.addWidget(self.prev_button)
        top_bar.addWidget(self.next_button)
        top_bar.addWidget(self.save_smi_button)
        top_bar.addWidget(self.save_csv_button)

        summary_row = QHBoxLayout()
        self.results_summary = QLabel("No results loaded")
        self.results_summary.setObjectName("SectionLabel")
        self.nav_label = QLabel("Page 1 of 1")
        self.nav_label.setObjectName("SmallInfoLabel")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter by SMILES substring")
        self.search_box.textChanged.connect(self.apply_filter)
        self.search_box.setMaximumWidth(360)

        summary_row.addWidget(self.results_summary)
        summary_row.addStretch(1)
        summary_row.addWidget(QLabel("Filter:"))
        summary_row.addWidget(self.search_box)
        summary_row.addWidget(self.nav_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_grid = QGridLayout(self.results_container)
        self.results_grid.setSpacing(14)
        self.scroll_area.setWidget(self.results_container)

        layout.addLayout(top_bar)
        layout.addLayout(summary_row)
        layout.addWidget(self.scroll_area)

    # ------------------------- Image and run controls -------------------------
    def select_image(self):
        supported_formats = "Image Files (*.png *.jpg *.jpeg *.bmp *.webp *.gif *.tiff *.tif)"
        start_dir = str(Path(self.image_path).parent) if self.image_path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", start_dir, supported_formats)
        if path:
            self.set_image(path)

    def set_image(self, path: str):
        self.image_path = path
        pixmap = pixmap_from_path(path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Image loading failed", "The selected image could not be previewed.")
            self.image_preview.setText("Preview not available")
        else:
            self.image_preview.setPixmap(
                pixmap.scaled(
                    self.image_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self.image_path_label.setText(path)
        self.start_button.setEnabled(True)
        self.open_results_button.hide()
        self.status_message.setText("Image loaded. Select a generation mode and start.")

    def reset_home_state(self):
        if self.is_running():
            QMessageBox.information(self, "Generation in progress", "Please wait until the current run has finished.")
            return
        self.image_path = None
        self.image_preview.clear()
        self.image_preview.setText("Drop-in preview\nSelect an image to start")
        self.image_path_label.setText("No image selected")
        self.start_button.setEnabled(False)
        self.open_results_button.hide()
        self.progress_bar.hide()
        self.progress_bar.setValue(0)
        self.status_message.setText("Ready")

    def open_new_image(self):
        if self.is_running():
            QMessageBox.information(self, "Generation in progress", "Please wait until the current run has finished.")
            return
        self.stacked_widget.setCurrentWidget(self.home_page)
        self.select_image()

    def is_running(self) -> bool:
        return bool(self.thread is not None and self.thread.isRunning())

    def start_analysis(self):
        if not self.image_path:
            QMessageBox.warning(self, "Missing image", "Please select an image first.")
            return
        if self.is_running():
            return

        self.results = []
        self.filtered_results = []
        self.image_cache = {}
        self.current_page = 0
        self.search_box.clear()

        generation_mode = self.mode_selector.currentData() or "hybrid"
        self.select_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.mode_selector.setEnabled(False)
        self.open_results_button.hide()
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status_message.setText(f"Running {self.mode_selector.currentText()}...")

        self.thread = QThread(self)
        self.worker = ImagichemWorker(self.image_path, generation_mode)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.analysis_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.thread_finished_cleanup)
        self.thread.start()

    def update_progress(self, value: int):
        value = max(0, min(int(value), 100))
        self.progress_bar.setValue(value)
        self.status_message.setText(f"Generating molecules... {value}%")

    def analysis_finished(self, results: list):
        self.results = results or []
        self.filtered_results = list(self.results)
        self.progress_bar.setValue(100)
        self.status_message.setText(f"Generation completed. {len(self.results)} molecules generated.")
        self.open_results_button.setText(f"Open {len(self.results)} Results")
        self.open_results_button.show()
        self.set_controls_enabled(True)

    def analysis_error(self, message: str):
        self.progress_bar.hide()
        self.status_message.setText("Generation failed.")
        self.set_controls_enabled(True)
        QMessageBox.critical(self, "Generation error", message)

    def thread_finished_cleanup(self):
        self.thread = None
        self.worker = None

    def set_controls_enabled(self, enabled: bool):
        self.select_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
        self.mode_selector.setEnabled(enabled)
        self.start_button.setEnabled(enabled and bool(self.image_path))

    # ------------------------- Result rendering -------------------------
    def get_mol_image(self, smiles: str) -> QPixmap:
        if smiles in self.image_cache:
            return self.image_cache[smiles]

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            pil_img = Draw.MolToImage(mol, size=(220, 220))
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            self.image_cache[smiles] = pixmap
            return pixmap

        blank = QPixmap(220, 220)
        blank.fill(Qt.GlobalColor.transparent)
        return blank

    def show_results_page(self):
        self.current_page = 0
        self.apply_filter()
        self.stacked_widget.setCurrentWidget(self.results_page)

    def apply_filter(self):
        query = self.search_box.text().strip().lower() if hasattr(self, "search_box") else ""
        if query:
            self.filtered_results = [(s, sc) for s, sc in self.results if query in s.lower()]
        else:
            self.filtered_results = list(self.results)
        self.current_page = 0
        self.display_page(self.current_page)

    def display_page(self, page_num: int):
        while self.results_grid.count():
            child = self.results_grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        total = len(self.filtered_results)
        total_pages = max(1, (total + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE)
        self.current_page = max(0, min(page_num, total_pages - 1))

        start = self.current_page * RESULTS_PER_PAGE
        end = min(start + RESULTS_PER_PAGE, total)

        mode_text = self.mode_selector.currentText()
        best = max((score for _, score in self.results), default=0.0)
        self.results_summary.setText(f"{total} shown / {len(self.results)} total | {mode_text} | Best score: {best:.2f}")
        self.nav_label.setText(f"Page {self.current_page + 1} of {total_pages}")

        self.prev_button.setEnabled(self.current_page > 0)
        self.next_button.setEnabled(self.current_page < total_pages - 1)
        self.save_smi_button.setEnabled(bool(self.results))
        self.save_csv_button.setEnabled(bool(self.results))

        if total == 0:
            empty = QLabel("No results match the current filter.")
            empty.setObjectName("SectionLabel")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.results_grid.addWidget(empty, 0, 0)
            return

        for local_idx, i in enumerate(range(start, end)):
            smiles, score = self.filtered_results[i]
            card = ResultCard(smiles, score, self.get_mol_image(smiles))
            row = local_idx // 4
            col = local_idx % 4
            self.results_grid.addWidget(card, row, col)

        self.scroll_area.verticalScrollBar().setValue(0)

    def next_page(self):
        self.display_page(self.current_page + 1)

    def prev_page(self):
        self.display_page(self.current_page - 1)

    # ------------------------- Export -------------------------
    def save_smiles(self):
        if not self.results:
            return
        default_name = self.default_output_name("smi")
        path, _ = QFileDialog.getSaveFileName(self, "Save SMILES", default_name, "SMILES Files (*.smi);;Text Files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                for smiles, _ in self.results:
                    handle.write(smiles + "\n")
            self.statusBar().showMessage(f"Saved {len(self.results)} SMILES to {path}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Save error", f"Could not save SMILES file:\n{exc}")

    def save_csv(self):
        if not self.results:
            return
        default_name = self.default_output_name("csv")
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", default_name, "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["rank", "smiles", "druglikeness_score", "generation_mode", "source_image"])
                for idx, (smiles, score) in enumerate(self.results, start=1):
                    writer.writerow([idx, smiles, f"{score:.6f}", self.mode_selector.currentData(), self.image_path or ""])
            self.statusBar().showMessage(f"Saved CSV to {path}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Save error", f"Could not save CSV file:\n{exc}")

    def default_output_name(self, extension: str) -> str:
        stem = "imagichem_output"
        if self.image_path:
            stem = Path(self.image_path).stem
        mode = self.mode_selector.currentData() or "hybrid"
        return f"{stem}_{mode}.{extension}"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
