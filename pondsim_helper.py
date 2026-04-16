import sys
import os
import traceback
import inspect
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import cv2 as cv
import netCDF4
import math
from typing import List, get_type_hints

from qgis.core import (QgsProject, QgsCoordinateReferenceSystem, QgsLayerTreeModel,
                       QgsRasterLayer, QgsStyle, QgsLayerTreeNode, QgsVectorLayer, QgsApplication, QgsAggregateCalculator, QgsField, QgsMeshLayer, QgsSingleBandPseudoColorRenderer, QgsColorRampShader, QgsRasterShader, QgsRasterBandStats, QgsPointXY, QgsFeature, QgsGeometry)
from qgis.gui import (QgsLayerTreeMapCanvasBridge, QgsLayerTreeView,
                      QgsDockWidget, QgsMapToolZoom, QgsMapToolPan, QgsSymbolSelectorDialog, QgisInterface)
from PyQt5 import (QtWidgets, QtCore)
from PyQt5.QtWidgets import (
    QPushButton, QAction, QMessageBox, QDialog, QMenu, QApplication, QFileDialog)
from PyQt5.QtCore import Qt, QVariant
from PyQt5.QtGui import QIntValidator

from landlab import RasterModelGrid
from landlab.io import read_esri_ascii, write_esri_ascii
from landlab.components.overland_flow import OverlandFlow
from landlab.components.sink_fill import SinkFiller
from landlab.components import FlowAccumulator, TrickleDownProfiler
from landlab.grid.raster_mappers import map_mean_of_horizontal_links_to_node, map_mean_of_vertical_links_to_node
from landlab.grid.mappers import map_value_at_max_node_to_link

# from PyQt5 import (QAxContainer, Qt, QtBluetooth, QtCore, QtDBus, QtDesigner, QtGui, QtHelp, QtLocation, QtMultimedia, QtMultimediaWidgets, QtNetwork, QtOpenGL, QtPositioning, QtPrintSupport, QtQml, QtQuick,
#                    QtQuick3D, QtQuickWidgets, QtRemoteObjects, QtSensors, QtSerialPort, QtSql, QtSvg, QtTest, QtTextToSpeech, QtWebChannel, QtWebKit, QtWebKitWidgets, QtWebSockets, QtWidgets, QtWinExtras, QtXml, QtXmlPatterns)

# from PyQt5.QAxContainer import *
# from PyQt5.Qt import QApplication
# from PyQt5.QtBluetooth import *
# from PyQt5.QtCore import *
# from PyQt5.QtDBus import *
# from PyQt5.QtDesigner import *
# from PyQt5.QtGui import *
# from PyQt5.QtHelp import *
# from PyQt5.QtLocation import *
# from PyQt5.QtMultimedia import *
# from PyQt5.QtMultimediaWidgets import *
# from PyQt5.QtNetwork import *
# from PyQt5.QtOpenGL import *
# from PyQt5.QtPositioning import *
# from PyQt5.QtPrintSupport import *
# from PyQt5.QtQml import *
# from PyQt5.QtQuick import *
# from PyQt5.QtQuick3D import *
# from PyQt5.QtQuickWidgets import *
# from PyQt5.QtRemoteObjects import *
# from PyQt5.QtSensors import *
# from PyQt5.QtSerialPort import *
# from PyQt5.QtSql import *
# from PyQt5.QtSvg import *
# from PyQt5.QtTest import *
# from PyQt5.QtTextToSpeech import *
# from PyQt5.QtWebChannel import *
# from PyQt5.QtWebKit import *
# from PyQt5.QtWebKitWidgets import *
# from PyQt5.QtWebSockets import *
# from PyQt5.QtWidgets import QMessageBox, QDialog, QMenu, QPushButton, QAction
# from PyQt5.QtWinExtras import *
# from PyQt5.QtXml import *
# from PyQt5.QtXmlPatterns import *
# from PyQt5.Qsci import *


# from PyQt5.QtWidgets import (
#     QPushButton, QAction, QMessageBox, QDialog, QMenu, QApplication)
# from PyQt5.QtCore import Qt

# # Explicit imports for PyInstaller hooks
from PyQt5 import (QtXml, QtNetwork, QtSql, QtGui,
                   QtPrintSupport, QtPositioning, sip, Qsci)
# from PyQt5 import (QtBluetooth, QtDBus, QtDesigner, QtHelp, QtLocation, QtMultimedia, QtMultimediaWidgets, QtOpenGL, QtQml, QtQuick, QtQuick3D, QtQuickWidgets, QtRemoteObjects, QtSensors, QtSerialPort, QtSvg, QtTest, QtTextToSpeech, QtWebChannel, QtWebKit, QtWebKitWidgets, QtWebSockets, QtWinExtras, QtXmlPatterns)

strMajorRelease = "0"
strMinorRelease = "0"
strUpdate = "0"
strVersion = strMajorRelease + "." + strMinorRelease + "." + strUpdate


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
        # res_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
        # res_path = os.path.join(base_path, relative_path)

    # return res_path
    return os.path.join(base_path, relative_path)


def find_module(obj):
    module = inspect.getmodule(obj)
    if module is None:
        return "Module not found"
    else:
        return module.__name__
