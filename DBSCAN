import sys
from PyQt5.QtWidgets import QApplication, QPushButton, QTextEdit, QWidget, QGridLayout, QLabel, QSpinBox, QDoubleSpinBox
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from math import hypot
import numpy as np
from itertools import cycle


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.build()

    def build(self):

        self.resize(1080, 760)
        self.setWindowTitle('Реализация алгоритма DBSCAN')
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.NEdit = QSpinBox()
        self.EPSEdit = QDoubleSpinBox()
        self.MEdit = QSpinBox()
        self.textEdit = QTextEdit()
        self.textEdit.setText('Рекомендуемые значения для N = 99\n'
                              'Esp = 0.2\n'
                              'M = 3-4')

        self.LabelN = QLabel('Введите N')
        self.LabelESP = QLabel('Введите eps')
        self.LabelM = QLabel('Введите m')

        self.button = QPushButton('Начать кластеризацию')
        self.buttonClear = QPushButton('Очистить окно')
        self.button.clicked.connect(self.plot)
        self.buttonClear.clicked.connect(self.ClearDialog)
        self.grid = QGridLayout()

        self.grid.addWidget(self.textEdit, 0, 0)

        self.grid.addWidget(self.LabelN, 1, 0)
        self.grid.addWidget(self.NEdit, 1, 1)

        self.grid.addWidget(self.LabelESP, 2, 0)
        self.grid.addWidget(self.EPSEdit, 2, 1)

        self.grid.addWidget(self.LabelM, 3, 0)
        self.grid.addWidget(self.MEdit, 3, 1)

        self.grid.addWidget(self.button, 4, 0)
        self.grid.addWidget(self.buttonClear, 4, 1)

        self.grid.addWidget(self.canvas, 0, 2, 4, 4)
        self.grid.addWidget(self.toolbar, 5, 2, 5, 4)

        self.setLayout(self.grid)

    def ClearDialog(self):
        self.NEdit.value(0)
        self.EPSEdit.value(0)
        self.MEdit.value(0)


    def plot(self):
        self.figure.clear()
        ax1 = self.figure.add_subplot(111)

        def dbscan_naive(list, eps, m, distance):
            NOISE = 0
            C = 0

            visited_points = set()
            clustered_points = set()
            clusters = {NOISE: []}

            def region_query(p):
                return [q for q in list if distance(p, q) < eps]

            def expand_cluster(p, neighbours):
                if C not in clusters:
                    clusters[C] = []
                clusters[C].append(p)
                clustered_points.add(p)
                while neighbours:
                    q = neighbours.pop()
                    if q not in visited_points:
                        visited_points.add(q)
                        neighbourz = region_query(q)
                        if len(neighbourz) > m:
                            neighbours.extend(neighbourz)
                    if q not in clustered_points:
                        clustered_points.add(q)
                        clusters[C].append(q)
                        if q in clusters[NOISE]:
                            clusters[NOISE].remove(q)

            for p in list:
                if p in visited_points:
                    continue
                visited_points.add(p)
                neighbours = region_query(p)
                if len(neighbours) < m:
                    clusters[NOISE].append(p)
                else:
                    C += 1
                    expand_cluster(p, neighbours)
            return clusters

        N = self.NEdit.value()
        eps = self.EPSEdit.value()
        m = self.MEdit.value()
        N1 = int(round(N / 3, 0))
        list = ([(np.random.randn() / 5 + 1, np.random.randn() / 5 + 1) for i in range(N1)])
        list.extend([(np.random.randn() / 10 + 5, np.random.randn() / 5 + 1) for i in range(N1)])
        list.extend([(np.random.randn() / 2 + 1, np.random.randn() / 3 + 1) for i in range(N1)])
        clusters = dbscan_naive(list, eps, m, lambda x, y: hypot(x[0] - y[0], x[1] - y[1]))
        for c, points in zip(cycle('bgrcmykgrcmykgrcmykgrcmykgrcmykgrcmyk'), clusters.values()):
            X = ([p[0] for p in points])
            Y = ([p[1] for p in points])
            ax1.scatter(X,Y)
        ax1.set_title('Алгоритм DBSCAN')
        ax1.set_ylabel('')
        ax1.set_xlabel('')
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
