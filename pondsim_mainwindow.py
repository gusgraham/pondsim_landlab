from pondsim_helper import *
from pondsim_projection_dialog import *
from pondsim_project_dialog import *

from ui_elements.Ui_pondsim_mainwindow_base import Ui_MainWindow


class gis_mainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    thisQgsProjectGPKGFileSpec = ''
    thisQgsProject = None
    thisQgsLayerTreeModel = None
    thisQgsLayerTreeView = None
    thisQgsLayerTree = None
    thisQgsLayerTreeMapCanvasBridge = None

    statusbarCrsButton = None
    statusbarCoordinates = None

    mainFlowbotDockWidget = None
    myMainWindow = None

    worldStreetMapLayer = None
    worldImageryLayer = None

    currentMapTool = None

    lastOpenDialogPath = ""

    def __init__(self, parent=None):
        """Constructor."""
        super(gis_mainWindow, self).__init__(parent)

        self.setupUi(self)

        self.statusbarCoordinates = QtWidgets.QPlainTextEdit()
        self.statusbarCoordinates.setFixedSize(220, 21)
        self.statusbarCoordinates.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff)
        self.statusbarCoordinates.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff)
        self.statusbarCoordinates.setLineWrapMode(
            QtWidgets.QPlainTextEdit.NoWrap)
        self.statusbarCoordinates.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByKeyboard | QtCore.Qt.TextSelectableByMouse)
        # self.plainTextEdit_2.setObjectName("plainTextEdit_2")

        # self.statusbarCoordinates = QLabel("Coordinates: ")
        self.mainStatusBar.addPermanentWidget(self.statusbarCoordinates)
        self.statusbarCrsButton = QPushButton("CRS: Unknown")
        self.statusbarCrsButton.clicked.connect(self.selectCrs)
        self.mainStatusBar.addPermanentWidget(self.statusbarCrsButton)

        self.thisQgsProject = QgsProject.instance()
        self.thisQgsProject.crsChanged.connect(self.updateCrsButton)

        self.thisQgsProject.setCrs(QgsCoordinateReferenceSystem("EPSG:27700"))

        self.mainCanvasWidget.setProject(self.thisQgsProject)
        self.mainCanvasWidget.setCanvasColor(QtCore.Qt.white)
        self.mainCanvasWidget.enableAntiAliasing(True)
        self.mainCanvasWidget.setDestinationCrs(self.thisQgsProject.crs())
        self.mainCanvasWidget.xyCoordinates.connect(self.updateCoordinates)

        aMenu = self.mainMenuBar.addMenu("File")

        action = QAction(u"New Project", self)
        action.triggered.connect(self.hello)
        aMenu.addAction(action)

        action = QAction(u"Open Project", self)
        action.triggered.connect(self.hello)
        aMenu.addAction(action)

        action = QAction(u"Save Project", self)
        action.triggered.connect(self.hello)
        aMenu.addAction(action)

        aMenu.addSeparator()

        action = QAction(u"Info", self)
        action.triggered.connect(self.aboutBox)
        aMenu.addAction(action)

        aMenu = self.mainMenuBar.addMenu("Layer")
        aSubMenu = aMenu.addMenu("Add Layer")

        action = QAction(u"Add Vector Layer...", self)
        action.triggered.connect(self.addVectorLayer)
        aSubMenu.addAction(action)

        aMenu = self.mainMenuBar.addMenu("Simulate")
        # aSubMenu = aMenu.addMenu("Run Sim")

        action = QAction(u"Run Simulation...", self)
        action.triggered.connect(self.runSimulation)
        aMenu.addAction(action)

        # Connect functions to the signals
        self.actionToggle_World_Street_Map.triggered.connect(
            self.toggleWorld_Street_Map)
        self.actionToggle_World_Imagery.triggered.connect(
            self.toggleWorld_Imagery)

        self.actionZoomIn.triggered.connect(self.setZoomInTool)
        self.actionZoomOut.triggered.connect(self.setZoomOutTool)
        self.actionPanMap.triggered.connect(self.setPanMapTool)
        self.actionZoomFull.triggered.connect(
            self.mainCanvasWidget.zoomToFullExtent)
        self.actionZoomPrevious.triggered.connect(self.zoomToPreviousExtent)
        self.actionZoomNext.triggered.connect(self.zoomToNextExtent)

        self.myMainWindow = self.centralWidget.parent()

        # Create Layer widget
        self.thisQgsLayerTree = self.thisQgsProject.layerTreeRoot()
        self.thisQgsLayerTreeMapCanvasBridge = QgsLayerTreeMapCanvasBridge(
            self.thisQgsLayerTree, self.mainCanvasWidget)
        self.thisQgsLayerTreeModel = QgsLayerTreeModel(self.thisQgsLayerTree)
        self.thisQgsLayerTreeModel.setFlag(QgsLayerTreeModel.AllowNodeReorder)
        self.thisQgsLayerTreeModel.setFlag(QgsLayerTreeModel.AllowNodeRename)
        self.thisQgsLayerTreeModel.setFlag(
            QgsLayerTreeModel.AllowNodeChangeVisibility)
        self.thisQgsLayerTreeModel.setFlag(QgsLayerTreeModel.ShowLegend)
        self.thisQgsLayerTreeView = QgsLayerTreeView()
        self.thisQgsLayerTreeView.setModel(self.thisQgsLayerTreeModel)
        self.mainQgsLayersDockWidget = QgsDockWidget(
            "Layers", self.myMainWindow)
        self.mainQgsLayersDockWidget.setObjectName("mainQgsLayersDockWidget")
        self.mainQgsLayersDockWidget.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFloatable | QtWidgets.QDockWidget.DockWidgetMovable)
        self.mainQgsLayersDockWidget.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.mainQgsLayersDockWidget.setWidget(self.thisQgsLayerTreeView)
        self.myMainWindow.addDockWidget(
            Qt.LeftDockWidgetArea, self.mainQgsLayersDockWidget)
        self.thisQgsLayerTreeView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.thisQgsLayerTreeView.customContextMenuRequested[QtCore.QPoint].connect(
            self.openQgsLayerTreeViewContextMenu)

    def zoomFullExtent(self):
        if not self.mainCanvasWidget.fullExtent() is None:
            self.mainCanvasWidget.setExtent(self.mainCanvasWidget.fullExtent())
            self.refresh()

    def aboutBox(self):

        msg = "Basic GIS " + strVersion + "\n" + "\n" + \
            "Contact: Fergus.Graham@rpsgroup.com"
        QMessageBox.information(None, 'About', msg, QMessageBox.Ok)

    def zoomToNextExtent(self):
        self.mainCanvasWidget.zoomToNextExtent()

    def zoomToPreviousExtent(self):
        self.mainCanvasWidget.zoomToPreviousExtent()

    def setZoomInTool(self):
        if self.actionZoomIn.isChecked() == True:
            self.currentMapTool = QgsMapToolZoom(
                self.mainCanvasWidget, False)  # To zoom in
            self.mainCanvasWidget.setMapTool(self.currentMapTool)
        else:
            if self.mainCanvasWidget.mapTool() == self.currentMapTool:
                self.mainCanvasWidget.unsetMapTool(self.currentMapTool)
                self.currentMapTool = None

    def setZoomOutTool(self):
        if self.actionZoomOut.isChecked() == True:
            self.currentMapTool = QgsMapToolZoom(
                self.mainCanvasWidget, True)  # To zoom out
            self.mainCanvasWidget.setMapTool(self.currentMapTool)
        else:
            if self.mainCanvasWidget.mapTool() == self.currentMapTool:
                self.mainCanvasWidget.unsetMapTool(self.currentMapTool)
                self.currentMapTool = None

    def setPanMapTool(self):
        if self.actionPanMap.isChecked() == True:
            self.currentMapTool = QgsMapToolPan(
                self.mainCanvasWidget)  # To pan
            self.mainCanvasWidget.setMapTool(self.currentMapTool)
        else:
            if self.mainCanvasWidget.mapTool() == self.currentMapTool:
                self.mainCanvasWidget.unsetMapTool(self.currentMapTool)
                self.currentMapTool = None

    def addVectorLayer(self):
        pass

    def updateCrsButton(self):
        # self.statusbarCrsButton.setText("CRS: " + self.thisQgsProject.crs().userFriendlyIdentifier())
        self.mainCanvasWidget.setDestinationCrs(self.thisQgsProject.crs())
        self.statusbarCrsButton.setText(
            "CRS: " + self.thisQgsProject.crs().authid())

    def selectCrs(self):
        selectProjectionDlg = pondsim_projectionDialog()
        selectProjectionDlg.setWindowTitle("Select CRS Dialog")
        selectProjectionDlg.mQgsProjectionSelectionWidget.setCrs(
            self.thisQgsProject.crs())
        ret = selectProjectionDlg.exec()
        if ret == QDialog.Accepted:
            self.thisQgsProject.setCrs(
                selectProjectionDlg.mQgsProjectionSelectionWidget.crs())
            self.mainCanvasWidget.setDestinationCrs(self.thisQgsProject.crs())

    def updateCoordinates(self, pointXY):
        self.statusbarCoordinates.setPlainText(
            "Coordinates: " + str(round(pointXY.x(), 4)) + ", " + str(round(pointXY.y(), 4)))

    def refresh(self):
        if not self.thisQgsLayerTreeView.currentLayer() is None:
            self.thisQgsLayerTreeView.refreshLayerSymbology(
                self.thisQgsLayerTreeView.currentLayer().id())
        self.mainCanvasWidget.refresh()

    def toggleWorld_Street_Map(self):
        if self.actionToggle_World_Street_Map.isChecked() == True:
            myUrl = "url='https://server.arcgisonline.com/arcgis/rest/services/World_Street_Map/MapServer' layer='0'"
            self.worldStreetMapLayer = QgsRasterLayer(
                myUrl, "World Street Map", providerType="arcgismapserver")
            self.worldStreetMapLayer.setCrs(
                QgsCoordinateReferenceSystem("EPSG:3857"))
            if not self.worldStreetMapLayer.isValid():
                # print("Layer failed to load!")
                QMessageBox.warning(
                    None, '', 'Layer failed to load!', QMessageBox.Ok)
            else:
                self.thisQgsProject.addMapLayer(self.worldStreetMapLayer)
        else:
            self.thisQgsProject.removeMapLayer(self.worldStreetMapLayer)
            self.worldStreetMapLayer = None
        self.refresh()

    def toggleWorld_Imagery(self):
        if self.actionToggle_World_Imagery.isChecked() == True:
            myUrl = "url='https://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer' layer='0'"
            self.worldImageryLayer = QgsRasterLayer(
                myUrl, "World Imagery", providerType="arcgismapserver")
            self.worldImageryLayer.setCrs(
                QgsCoordinateReferenceSystem("EPSG:3857"))
            if not self.worldImageryLayer.isValid():
                # print("Layer failed to load!")
                QMessageBox.warning(
                    None, '', 'Layer failed to load!', QMessageBox.Ok)
            else:
                self.thisQgsProject.addMapLayer(self.worldImageryLayer)
        else:
            self.thisQgsProject.removeMapLayer(self.worldImageryLayer)
            self.worldImageryLayer = None
        self.refresh()
#
#
#

    def editSymbol(self):
        style = QgsStyle()
        mySymbolDlg = QgsSymbolSelectorDialog(
            self.thisQgsLayerTreeView.currentLayer().renderer().symbol(), style, self.thisQgsLayerTreeView.currentLayer(), self.thisQgsLayerTreeView)
        mySymbolDlg.setWindowTitle("Dialog")
        ret = mySymbolDlg.exec()
        if ret == QDialog.Accepted:
            self.thisQgsLayerTreeView.refreshLayerSymbology(
                self.thisQgsLayerTreeView.currentLayer().id())
            self.mainCanvasWidget.refresh()

        mySymbolDlg = None

    def openBrowserFile(self, fileName, fileTypeHint):
        ret = QMessageBox.warning(None, 'Warning', 'Open file', QMessageBox.Ok)

    def hello(self):

        ret = QMessageBox.warning(
            None, 'Warning', 'Hello World', QMessageBox.Ok)

    def openQgsLayerTreeViewContextMenu(self, position):

        # level = self.getTreeViewLevel(self.trwSummedFMs)
        if not self.thisQgsLayerTreeView.index2node(self.thisQgsLayerTreeView.indexAt(position)) is None:
            if self.thisQgsLayerTreeView.index2node(self.thisQgsLayerTreeView.indexAt(position)).nodeType() == QgsLayerTreeNode.NodeType.NodeLayer:
                if type(self.thisQgsLayerTreeView.index2node(self.thisQgsLayerTreeView.indexAt(position)).layer()) == QgsVectorLayer:
                    menu = QMenu()
                    myCallback = QtWidgets.QAction("Zoom to Layer", menu)
                    myCallback.triggered.connect(self.editSymbol)
                    menu.addAction(myCallback)

                    myCallback = QtWidgets.QAction("Edit Symbology", menu)
                    myCallback.triggered.connect(self.editSymbol)
                    menu.addAction(myCallback)

                    if not len(menu.actions()) == 0:
                        menu.exec_(
                            self.thisQgsLayerTreeView.mapToGlobal(position))

    def runSimulation(self):
        projectDlg = pondsim_projectDialog(self.thisQgsProject)
        projectDlg.setWindowTitle("Pondsim Settings")
        ret = projectDlg.exec()
        if ret == QDialog.Accepted:
            pass


def setup_qgis(qgs_app):
    # """ Set QGIS paths based on whether running as a bundled application or not """
    # # if getattr(sys, "frozen", False):
    # #     print("Running In An Application Bundle")
    bundle_dir = resource_path('qgis_stable\\Library')
    # bundle_dir_forward_slashes = bundle_dir.replace("\\", "/")
    qgis_prefix_path = bundle_dir
    qgis_plugin_path = bundle_dir + "\\plugins"

    # os.environ["OSGEO4W_ROOT"] = bundle_dir
    os.environ["GDAL_DATA"] = bundle_dir + "\\share\\gdal"
    os.environ["GDAL_DRIVER_PATH"] = bundle_dir + "\\lib\\gdalplugins"
    os.environ["GEOTIFF_CSV"] = bundle_dir + "\\share\\epsg_csv"
    os.environ["PDAL_DRIVER_PATH"] = bundle_dir + "\\bin"
    os.environ["QT_PLUGIN_PATH"] = bundle_dir + \
        "\\qtplugins;" + bundle_dir + "\\plugins"

    # # os.environ["GS_LIB"] = bundle_dir + "\\apps\\gs\\lib"
    # # os.environ["JPEGMEM"] = "1000000"
    # os.environ["GDAL_FILENAME_IS_UTF8"] = "YES"
    # os.environ["O4W_QT_BINARIES"] = bundle_dir_forward_slashes + \
    #     "/apps/Qt5/bin"
    # os.environ["O4W_QT_HEADERS"] = bundle_dir_forward_slashes + \
    #     "/apps/Qt5/include"
    # os.environ["O4W_QT_LIBRARIES"] = bundle_dir_forward_slashes + \
    #     "/apps/Qt5/lib"
    # os.environ["O4W_QT_PLUGINS"] = bundle_dir_forward_slashes + \
    #     "/apps/Qt5/plugins"
    # os.environ["O4W_QT_PREFIX"] = bundle_dir_forward_slashes + "/apps/Qt5"
    # os.environ["O4W_QT_TRANSLATIONS"] = bundle_dir_forward_slashes + \
    #     "/apps/Qt5/translations"
    # os.environ["PDAL_DRIVER_PATH"] = bundle_dir + "\\apps\\pdal\\plugins"
    # os.environ["PROJ_LIB"] = bundle_dir + "\\share\\proj"
    # os.environ["PYTHONHOME"] = bundle_dir + "\\apps\\Python39"
    # os.environ["PYTHONUTF8"] = "1"

    # os.environ["QT_PLUGIN_PATH"] = bundle_dir + "\\apps\\Qt5\\plugins"
    # os.environ["SSL_CERT_DIR"] = bundle_dir + "\\apps\\openssl\\certs"
    # os.environ["SSL_CERT_FILE"] = bundle_dir + "\\bin\\curl-ca-bundle.crt"
    # os.environ["VSI_CACHE"] = "TRUE"
    # os.environ["VSI_CACHE_SIZE"] = "1000000"

    # os.environ["PYTHONPATH"] = bundle_dir + "\\python;" + bundle_dir + "\\python\\plugins;" + \
    #     bundle_dir + "\\python\\qgis;" + bundle_dir + "\\bin;" + bundle_dir + "\\lib\\site-packages"

    sys.path.append(bundle_dir + "\\python")
    sys.path.append(bundle_dir + "\\python\\plugins")
    sys.path.append(bundle_dir + "\\python\\qgis")
    sys.path.append(bundle_dir + "\\bin")
    sys.path.append(bundle_dir + "\\lib\\site-packages")

    # os.environ["PATH"] = bundle_dir + \
    #     "\\bin;C:\\Windows\\system32;C:\\Windows;C:\\Windows\\system32\\WBem"

    # os.environ["O4W_QT_DOC"] = bundle_dir_forward_slashes + "/apps/Qt5/doc"
    # if os.path.exists(bundle_dir + "\\apps\\qgis-ltr\\bin\\qgisgrass7.dll"):
    #     os.environ["GISBASE"] = bundle_dir + "\\apps\\grass\\grass78"
    #     os.environ["GRASS_PYTHON"] = bundle_dir + "\\bin\\python3.exe"
    #     os.environ["GRASS_PROJSHARE"] = bundle_dir + "\\share\\proj"
    #     os.environ["FONTCONFIG_FILE"] = bundle_dir + \
    #         "\\apps\\grass\\grass78\\etc\\fonts.conf"

    # os.environ["QT_PLUGIN_PATH"] = bundle_dir + \
    #     "\\apps\\qgis-ltr\\qtplugins;" + bundle_dir + "\\apps\\qt5\\plugins"
    # if os.path.exists(bundle_dir + "\\apps\\qgis-ltr\\bin\\qgisgrass7.dll"):
    #     osPathString = bundle_dir + "\\apps\\qgis-ltr\\bin;" + bundle_dir + \
    #         "\\apps\\grass\\grass78\\lib;" + bundle_dir + "\\apps\\grass\\grass78\\bin;"
    # else:
    #     osPathString = bundle_dir + "\\apps\\qgis-ltr\\bin;"

    # osPathString = osPathString + bundle_dir + "\\apps\\qt5\\bin;" + \
    #     os.environ["PATH"] + \
    #     ";C:\\tempOSGeo4W\\bin;C:\\Windows\\system32;C:\\Windows;C:\\Windows\\system32\\WBem"
    # os.environ["PATH"] = osPathString

    qgs_app.setPrefixPath(qgis_prefix_path, True)
    # qgs_app.setPrefixPath(os.environ["QGIS_PREFIX_PATH"], True)
    qgs_app.setPluginPath(qgis_plugin_path)
    qgs_app.initQgis()


# example layer
# testlayer = r'C:\Temp\ATO_Permeable_Areas.shp'
app = QApplication(sys.argv)
qgs = QgsApplication([], True)
setup_qgis(qgs)

# qgs.setPrefixPath(resource_path('resources\\QGIS\\apps\\qgis'), True)
# # QMessageBox.information(None, 'TEST',
# #                         "\n".join(os.environ['TEST']), QMessageBox.Ok)
# QMessageBox.information(None, 'qgs.prefixPath()',
#                         qgs.prefixPath(), QMessageBox.Ok)
# QMessageBox.information(None, 'sys.path',
#                         "\n".join(sys.path), QMessageBox.Ok)
# qgs.initQgis()

mainWindow = gis_mainWindow()
mainWindow.setWindowTitle("Pondsim v" + strVersion)
mainWindow.show()

# keys_to_retrieve = ['PATH', 'PYTHONPATH', 'GDAL_DATA', 'GDAL_DRIVER_PATH', 'GDAL_FILENAME_IS_UTF8', 'PDAL_DRIVER_PATH',
#                     'OSGEO4W_ROOT', 'PROJ_LIB', 'PYTHONHOME', 'PYTHONUTF8', 'QGIS_PREFIX_PATH', 'QT_PLUGIN_PATH',
#                     'VSI_CACHE', 'VSI_CACHE_SIZE', 'O4W_QT_PREFIX', 'O4W_QT_BINARIES', 'O4W_QT_PLUGINS',
#                     'O4W_QT_LIBRARIES', 'O4W_QT_TRANSLATIONS', 'O4W_QT_HEADERS', 'QGIS_WIN_APP_NAME',
#                     'SSL_CERT_DIR', 'SSL_CERT_FILE']

# QMessageBox.information(None, 'qgs.prefixPath()',
#                         qgs.prefixPath(), QMessageBox.Ok)
# QMessageBox.information(None, 'sys.path',
#                         "\n".join(sys.path), QMessageBox.Ok)
# QMessageBox.information(None, 'os.path', "\n".join(
#     [f"{key}: {value}" for key, value in os.environ.items() if key in keys_to_retrieve]), QMessageBox.Ok)

if QtCore.QT_VERSION >= 0x50501:
    def excepthook(type_, value, traceback_):
        traceback.print_exception(type_, value, traceback_)
        QtCore.qFatal('')
sys.excepthook = excepthook

app.exec_()
