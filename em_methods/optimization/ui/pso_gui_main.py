from PyQt5.QtWidgets import QApplication, QMainWindow, QStyleFactory
from PyQt5.QtCore import QSysInfo
from em_methods.optimization.ui.pso_gui import Ui_MainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from typing import Any

import sys
import logging

class PltFigure(FigureCanvasQTAgg):
    """
    Class to draw canvas for a particular figure
    Args:
        parent: parent widget where the figure will be drawn
        xlabel, ylabel: Labels for x and yaxis
        width, height, dpi: Image size/resolution
    Properties:
    """
    def __init__(
            self, parent, xlabel, ylabel, width=2.5, height=9, dpi=100, interactive=False
        ):
            """Initialize all the figure main elements"""
            logging.info("Initialize Figure Canvas")
            self.xlabel = xlabel
            self.ylabel = ylabel
            self._fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes: Any = self._fig.add_subplot(111)
            self.draw_axes()
            super(PltFigure, self).__init__(self._fig)
            parent.addWidget(self)

    def draw_axes(self, xlabel=None, ylabel=None):
        """Update x and ylabels for the plot"""
        logging.debug("Draw/Labelling Axis")
        if xlabel:
            self.xlabel: str = xlabel
        if ylabel:
            self.ylabel: str = ylabel
        self.axes.yaxis.grid(True, linestyle="-")
        self.axes.xaxis.grid(True, linestyle="-")
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_xlabel(self.xlabel)

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow,self).__init__()
        logging.info("Initializing UI.\n") # TODO: Connect buttons to functionalities 
        # Set up the UI
        self.setWindowTitle("Particle Swarm Optimization GUI")
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Apply system style
        app_style = QStyleFactory.create('Fusion')
        if app_style and QSysInfo.windowsVersion() == QSysInfo.WV_WINDOWS10:
            QApplication.setStyle(app_style)
        # Get the system's palette
        app_palette = QApplication.palette()
        # Set the application palette to the system palette
        QApplication.setPalette(app_palette)

        #Initialize the Main Figure and toolbar
        logging.debug("Initializing variable aliases.")
        self.sum_graph_layout = self.ui.sum_graph_layout
        self.main_canvas = PltFigure(self.sum_graph_layout, "A somewhat long x label just to have fun.", "KERNEL PANIC. Jk, another example for testing.")

def init_gui():
#initializing PSO gui if condition is == true
    # Create a QApplication instance with command-line arguments
    app = QApplication(sys.argv)

    # Show the window
    window=MyWindow()
    window.show()

    # Start the application event loop
    sys.exit(app.exec_())