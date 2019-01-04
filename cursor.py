#!/usr/bin/python3
#
# By Sebastian Raaphorst, 2018
#
# A visual example of the dispersive fly optimization problem where the flies attempt to chase around
# the moving cursor.

import sys
from time import sleep
from math import sqrt
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt
import numpy as np
from dispersive_flies import DispersiveFlies, euclidean_metric


position = [0, 0]


def update(ex):
    def update_fn(flies):
        ex.flies = flies
        ex.update()
    return update_fn


class Cursor(QWidget):

    def __init__(self, width=800, height=800):
        self.flies = []
        super().__init__()
        self.setGeometry(100, 100, width, height)
        self.setWindowTitle('Dispersive Flies Optimization')
        self.show()
        self.setMouseTracking(True)
        self.df = DispersiveFlies(2, fitness, flies=50, dim_max=(800 * np.ones(2)),
                                  metric=euclidean_metric, end_round=update(self))

    def paintEvent(self, _):
        self.repaint()

    def repaint(self):
        qp = QPainter()
        qp.begin(self)
        size = self.size()
        qp.setPen(Qt.black)
        qp.setBrush(Qt.black)
        qp.drawRect(0, 0, size.width(), size.height())

        # Draw the flies.
        qp.setPen(Qt.white)
        qp.setBrush(Qt.white)
        for (x, y) in self.flies:
            qp.drawEllipse(x-2, y-2, 4, 4)
        qp.end()

    def mouseMoveEvent(self, event):
        self.df.run_round()
        position[0:2] = (event.x(), event.y())
        self.update()
        sleep(0.1)


def fitness(fly):
    px, py = position
    fx, fy = fly
    return -sqrt((px - fx)**2 + (py - fy)**2)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Cursor()
    sys.exit(app.exec_())
