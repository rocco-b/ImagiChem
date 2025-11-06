import sys
import os
import io
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QStackedWidget, QScrollArea, QGridLayout, QFrame,
                             QSizePolicy, QLineEdit, QStatusBar)
from PyQt6.QtGui import QPixmap, QMovie, QPainter, QColor, QFont, QIcon 
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QSize, QRunnable, QThreadPool, QTimer
from rdkit import Chem
from rdkit.Chem import Draw

import imagichem_core

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

STYLESHEET = """
QWidget {
    background-color: #ffffff;
    color: #313131;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13pt;
}
QMainWindow {
    background-color: #FF004D;
}
QPushButton {
    background-color: #454ADE;
    color: #ffffff;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
}
QPushButton:hover {
    background-color: #5b5fdc;
}
QPushButton:pressed {
    background-color: #abadf3;
}
QPushButton:disabled {
    background-color: #d3d3d3;
    color: #313131;
}
QLabel#TitleLabel {
    font-size: 24pt;
    font-weight: bold;
    color: #454ADE;
}
QLabel#ImagePreviewLabel {
    border: 2px dashed #b4b4bc;
    border-radius: 5px;
    background-color: #d7d7e3;
}
QLabel#FooterLabel {
    color: #313131;
    font-size: 10pt;
}
QScrollArea {
    border: none;
}
QLineEdit {
    background-color: #3B4252;
    border: 1px solid #4C566A;
    border-radius: 3px;
    padding: 5px;
    font-family: 'Courier New', monospace;
}
QFrame#CardFrame {
    border: 1px solid #c3c3c3;
    border-radius: 8px;
    background-color: #ffffff;
}
"""

RESULTS_PER_PAGE = 52

class ImagichemWorker(QObject):
    finished = pyqtSignal(list)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            results = imagichem_core.run_imagichem_processing(self.image_path, self.progress.emit)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(f"An unexpected error occurred: {str(e)}")


class ColorProgressBar(QWidget):
    def __init__(self, score=0.0):
        super().__init__()
        self.score = score
        self.setMinimumHeight(20)

    def set_score(self, score):
        self.score = score
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        painter.setBrush(QColor("#f6f8fa"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 5, 5)

        if self.score >= 99.99:
            color = QColor("#37b64d")
        elif self.score >= 90:
            color = QColor("#77d132")
        elif self.score >= 70:
            color = QColor("#ffe61d")
        elif self.score >= 50:
            color = QColor("#ff8815")
        else:
            color = QColor("#f81222")
        
        painter.setBrush(color)
        width = int(self.width() * (self.score / 100.0))
        painter.drawRoundedRect(0, 0, width, self.height(), 5, 5)

        painter.setPen(QColor("#313131"))
        font = QFont('Segoe UI', 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{self.score:.2f}%")

class ScoreBar(QWidget):
    def __init__(self, score=0.0, score_range=(0, 100), text_format="{:.2f}", color_map=None):
        super().__init__()
        self.score = score
        self.score_range = score_range
        self.text_format = text_format
        self.color_map = color_map if color_map else {}
        self.setMinimumHeight(20)

    def set_score(self, score):
        self.score = score
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        painter.setBrush(QColor("#f6f8fa"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 5, 5)

        color = self.color_map.get(self.score, QColor("#37b64d"))
        painter.setBrush(color)
        
        score_min, score_max = self.score_range
        normalized_score = (self.score - score_min) / (score_max - score_min)
        width = int(self.width() * normalized_score)
        painter.drawRoundedRect(0, 0, width, self.height(), 5, 5)

        painter.setPen(QColor("#313131"))
        font = QFont('Segoe UI', 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text_format.format(self.score))

class ResultCard(QFrame):
    def __init__(self, smiles, sa_score, pains_match, image_pixmap):
        super().__init__()
        self.setObjectName("CardFrame")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.smiles_string = smiles

        layout = QVBoxLayout(self)
        
        self.image_label = QLabel()
        self.image_label.setPixmap(image_pixmap)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(250, 250)

        self.copy_button = QPushButton("Copy SMILES")
        self.copy_button.clicked.connect(self.copy_smiles_to_clipboard)

        sa_label = QLabel("Synthetic Accessibility (SA):")
        sa_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sa_bar = ScoreBar(score=sa_score, score_range=(1, 10), text_format="{:.2f}", color_map=self.get_sa_color_map(sa_score))

        pains_label = QLabel("PAINS:")
        pains_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pains_score = 0 if pains_match == "None" else 1
        self.pains_bar = ScoreBar(score=pains_score, score_range=(0, 1), text_format=pains_match, color_map=self.get_pains_color_map(pains_score))

        layout.addWidget(self.image_label)
        layout.addWidget(self.copy_button)
        layout.addSpacing(10)
        layout.addWidget(sa_label)
        layout.addWidget(self.sa_bar)
        layout.addSpacing(5)
        layout.addWidget(pains_label)
        layout.addWidget(self.pains_bar)
    def get_sa_color_map(self, score):
        return {score: QColor("#37b64d") if score <= 3 else (QColor("#ffe61d") if score <= 7 else QColor("#f81222"))}
    def get_pains_color_map(self, score):
        return {0: QColor("#37b64d"), 1: QColor("#f81222")}

    def copy_smiles_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.smiles_string)
        
        self.copy_button.setText("Copied!")
        QTimer.singleShot(2000, lambda: self.copy_button.setText("Copy SMILES"))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImagiChem - Imagine Chemistry")

        self.setWindowIcon(QIcon(resource_path("icon.ico")))


        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(STYLESHEET)

        self.image_path = None
        self.results = []
        self.image_cache = {}
        self.current_page = 0

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        self.init_page1()
        self.init_page2()

        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)
        
        self.init_footer()

    def init_footer(self):
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        footer_label = QLabel("Designed and developed by Prof. Antonio Rescifina and Dr. Rocco Buccheri at Catania University")
        footer_label.setObjectName("FooterLabel")
        footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_bar.addWidget(footer_label, 1)

    def init_page1(self):
        self.page1 = QWidget()
        layout = QVBoxLayout(self.page1)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("ImagiChem")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.image_preview = QLabel("Select an image to start")
        self.image_preview.setObjectName("ImagePreviewLabel")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setFixedSize(400, 400)

        self.select_button = QPushButton("Select Image")
        self.select_button.clicked.connect(self.select_image)
        
        self.start_button = QPushButton("Start Analysis")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_analysis)
        
        self.loading_label = QLabel()
        self.loading_movie = QMovie("loading.gif")
        self.loading_movie.setScaledSize(QSize(50, 50))
        self.loading_label.setMovie(self.loading_movie)
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.hide()
        
        self.results_button = QPushButton("Check Results")
        self.results_button.clicked.connect(self.show_results_page)
        self.results_button.hide()
        
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(self.image_preview)
        layout.addSpacing(20)
        layout.addWidget(self.select_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.loading_label)
        layout.addWidget(self.results_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()

    def init_page2(self):
        self.page2 = QWidget()
        layout = QVBoxLayout(self.page2)
        
        top_bar = QHBoxLayout()
        
        self.nav_label = QLabel("Page 1")
        prev_button = QPushButton("< Prev")
        prev_button.clicked.connect(self.prev_page)
        next_button = QPushButton("Next >")
        next_button.clicked.connect(self.next_page)
        save_button = QPushButton("Save All SMILES")
        save_button.clicked.connect(self.save_results)
        
        top_bar.addSpacing(20)
        top_bar.addWidget(prev_button)
        top_bar.addWidget(self.nav_label, alignment=Qt.AlignmentFlag.AlignCenter)
        top_bar.addWidget(next_button)
        top_bar.addStretch()
        top_bar.addWidget(save_button)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_grid = QGridLayout(self.results_container)
        self.scroll_area.setWidget(self.results_container)
        
        layout.addLayout(top_bar)
        layout.addWidget(self.scroll_area)

    def select_image(self):
            supported_formats = "Image Files (*.png *.jpg *.jpeg *.bmp *.webp *.gif *.tiff)"
            path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", supported_formats)
            
            if path:
                self.image_path = path
                pixmap = QPixmap(path)
                self.image_preview.setPixmap(pixmap.scaled(self.image_preview.size(), 
                                                        Qt.AspectRatioMode.KeepAspectRatio, 
                                                        Qt.TransformationMode.SmoothTransformation))
                self.start_button.setEnabled(True)
                self.results_button.hide()

    def start_analysis(self):
        self.select_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.loading_label.show()
        self.loading_movie.start()

        self.thread = QThread()
        self.worker = ImagichemWorker(self.image_path)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.analysis_error)
        self.worker.progress.connect(lambda p: self.start_button.setText(f"Analyzing... {p}%"))

        self.thread.start()

    def analysis_finished(self, results):
        self.results = results
        self.loading_movie.stop()
        self.loading_label.hide()
        self.start_button.setText("Analysis Complete")
        self.results_button.setText(f"Check {len(results)} Results")
        self.results_button.show()
        
        self.thread.quit()
        self.thread.wait()
        
        self.preload_images_for_page(0)

    def analysis_error(self, err_msg):
        self.loading_movie.stop()
        self.loading_label.hide()
        self.start_button.setText("Error!")
        self.start_button.setEnabled(True)
        self.select_button.setEnabled(True)
        print(err_msg)
    
    def get_mol_image(self, smiles):
        if smiles in self.image_cache:
            return self.image_cache[smiles]
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            pil_img = Draw.MolToImage(mol, size=(250, 250))
            
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())

            self.image_cache[smiles] = pixmap
            return pixmap
            
        blank_pixmap = QPixmap(250, 250)
        blank_pixmap.fill(Qt.GlobalColor.transparent)
        return blank_pixmap

    def preload_images_for_page(self, page_num):
        start_index = page_num * RESULTS_PER_PAGE
        end_index = min(start_index + RESULTS_PER_PAGE, len(self.results))
        
        for i in range(start_index, end_index):
            smiles = self.results[i][0]
            if smiles not in self.image_cache:
                 self.get_mol_image(smiles)

    def show_results_page(self):
        self.current_page = 0
        self.display_page(self.current_page)
        self.stacked_widget.setCurrentWidget(self.page2)
        if len(self.results) > RESULTS_PER_PAGE:
             QThreadPool.globalInstance().start(lambda: self.preload_images_for_page(1))
    
    def display_page(self, page_num):
        while self.results_grid.count():
            child = self.results_grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        start_index = page_num * RESULTS_PER_PAGE
        end_index = min(start_index + RESULTS_PER_PAGE, len(self.results))
        
        if start_index >= len(self.results):
            return

        total_pages = (len(self.results) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
        self.nav_label.setText(f"Page {page_num + 1} of {total_pages}")
        
        for i in range(start_index, end_index):
            smiles, sa_score, pains_match = self.results[i]
            image_pixmap = self.get_mol_image(smiles)
            
            card = ResultCard(smiles, sa_score, pains_match, image_pixmap)
            
            row = (i - start_index) // 4
            col = (i - start_index) % 4
            self.results_grid.addWidget(card, row, col)

        self.scroll_area.verticalScrollBar().setValue(0)

    def next_page(self):
        total_pages = (len(self.results) -1) // RESULTS_PER_PAGE
        if self.current_page < total_pages:
            self.current_page += 1
            self.display_page(self.current_page)
            if (self.current_page + 1) <= total_pages:
                 QThreadPool.globalInstance().start(lambda: self.preload_images_for_page(self.current_page + 1))


    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.display_page(self.current_page)

    def save_results(self):
        if not self.results:
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Results", "imagichem_results.csv", "CSV Files (*.csv)")
        
        if path:
            try:
                with open(path, 'w') as f:
                    f.write("SMILES,SA_Score,PAINS_Match\n")
                    
                    for smiles, sa, pains in self.results:
                        f.write(f'{smiles},{sa:.4f},"{pains}"\n')
            except Exception as e:
                print(f"Error saving file: {e}")

    def return_to_main_page(self):
        self.image_path = None
        self.results = []
        self.image_cache = {}
        self.current_page = 0

        self.image_preview.clear()
        self.image_preview.setText("Select an image to start")
        
        self.select_button.setEnabled(True)
        
        self.start_button.setEnabled(False)
        self.start_button.setText("Start Analysis")
        
        self.results_button.hide()
        
        self.stacked_widget.setCurrentWidget(self.page1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())