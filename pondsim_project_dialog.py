from pondsim_helper import *

from ui_elements.Ui_pondsim_project_dialog_base import Ui_Dialog


class pondsim_projectDialog(QtWidgets.QDialog, Ui_Dialog):

    def __init__(self, thisProject, parent=None):
        """Constructor."""
        super(pondsim_projectDialog, self).__init__(parent)
        self.setupUi(self)

        self.thisProject: QgsProject = thisProject
        self.groundModelQGISLayer: QgsRasterLayer = None
        self.groundModelQGISLayer_layerid: str = ''
        self.groundModelRMG: RasterModelGrid = None
        self.groundModelElevationData = None
        self.groundModelRMG_Filled: RasterModelGrid = None
        self.groundModelElevationData_Filled = None
        self.surfaceWaterDepthResults = None
        self.surfaceWaterLevelResults = None
        self.exportNetCDF: bool = False
        self.useVariableTimestep: bool = True
        self.useUH: bool = True
        self.steepSlopes: bool = True

        self.pointSourceVectorLayer: QgsVectorLayer = None
        self.pointSourceVectorLayer_layerid = ''
        self.surfaceWaterDepthResults: QgsRasterLayer = None
        self.surfaceWaterDepthResults_layerid = ''
        self.surfaceWaterLevelResults: QgsRasterLayer = None
        self.surfaceWaterLevelResults_layerid = ''

        self.timeToPeak_s = 3600
        self.maxNodeVol_m3: float = 0
        self.maxNodePeakFlow_m3s: float = 0
        self.maxNodeAvgFlow_m3s: float = 0
        # self.longestHydrographTime_s: float = 0
        self.hydrographDuration_s: float = 3600 * 5
        self.additionalLag_s: float = 0
        self.simulationDuration_s: float = 3600 * 5
        self.fixedTimestep_s: float = 60
        self.animationTimestep_s: float = 300
        self.totalInflowVolume_m3: float = 0
        self.currentInflowVolume_m3: float = 0

        self.maxDepthThreshold: float = 0.00005

        self.nodeHydrographs: pd.DataFrame = None

        validator = QIntValidator()
        self.txtAdditionalLag.setValidator(validator)
        # self.dblSpnTimeToPeak.valueChanged.connect(self.updateSimulationStats)
        self.dblSpnStormDuration.valueChanged.connect(
            self.updateSimulationStats)
        self.spnMaxTimestep.valueChanged.connect(self.updateSimulationStats)
        self.spnAnimationTimestep.valueChanged.connect(
            self.updateSimulationStats)
        self.chkCreateNetCDF.toggled.connect(self.enableButtons)
        self.chkUseVariableTimestep.toggled.connect(self.enableButtons)
        self.chkSteepSlopes.toggled.connect(self.enableButtons)
        self.rbnNCRS.toggled.connect(self.volumeMethodChanged)
        self.rbnFlat.toggled.connect(self.volumeMethodChanged)

        self.currentlySimming: bool = False

        self.btnOpenRasterLayer.clicked.connect(self.openGroundModelRaster)
        self.btnCreatePointSourceQGISLayer.clicked.connect(
            self.createPointSourceVectorLayer)
        self.btnSelectOutputFolder.clicked.connect(self.selectOutputFolder)
        # self.btnRunOverlandFlow.clicked.connect(self.profileRunOverlandFlowAnalysis)
        self.btnRunOverlandFlow.clicked.connect(self.runOverlandFlowAnalysis)
        self.btnCreateLowResGrid.clicked.connect(self.createLowResGrid)

        self.thisProject.layerWillBeRemoved.connect(self.onLayerWillBeRemoved)
        # QgsProject.instance().layerWillBeRemoved.connect(self.onLayerWillBeRemoved)

        self.btnOK.clicked.connect(self.onAccept)
        self.btnCancel.clicked.connect(self.onReject)

    def onAccept(self):
        self.accept()

    def onReject(self):
        self.reject()

    def volumeMethodChanged(self):
        self.useUH = self.rbnNCRS.isChecked()
        self.updateSimulationStats()

    def enableButtons(self):
        self.dblSpnStormDuration.setEnabled(not self.currentlySimming)
        self.txtAdditionalLag.setEnabled(not self.currentlySimming)
        self.spnMaxTimestep.setEnabled(not self.currentlySimming)
        self.spnAnimationTimestep.setEnabled(not self.currentlySimming)
        self.btnRunOverlandFlow.setEnabled(not self.currentlySimming)
        if not self.currentlySimming:
            self.label_11.setEnabled(self.chkCreateNetCDF.isChecked())
            self.spnAnimationTimestep.setEnabled(
                self.chkCreateNetCDF.isChecked())
            self.exportNetCDF = self.chkCreateNetCDF.isChecked()
            self.useVariableTimestep = self.chkUseVariableTimestep.isChecked()
            if self.useVariableTimestep:
                self.lblAdaptiveTS.setText('Adaptive Timestep (s):')
                self.spnMaxTimestep.setValue(0)
                self.spnMaxTimestep.setEnabled(False)
            else:
                self.lblAdaptiveTS.setText('Fixed Timestep (s):')
                self.spnMaxTimestep.setValue(self.fixedTimestep_s)
                self.spnMaxTimestep.setEnabled(True)
            self.steepSlopes = self.chkSteepSlopes.isChecked()

    def onLayerWillBeRemoved(self, layer_id):
        if layer_id == self.groundModelQGISLayer_layerid:
            self.groundModelQGISLayer = None
            self.groundModelQGISLayer_layerid = ''
            self.groundModelRMG = None
            self.groundModelElevationData = None
            self.groundModelRMG_Filled = None
            self.groundModelElevationData_Filled = None
            self.txtGroundModelQGISLayer.setText("")
        if layer_id == self.pointSourceVectorLayer_layerid:
            self.pointSourceVectorLayer = None
            self.pointSourceVectorLayer_layerid = ''
            self.txtPointSourceQGISLayer.setText("")
            self.updateSimulationStats()
        if layer_id == self.surfaceWaterDepthResults_layerid:
            self.surfaceWaterLevelResults = None
            self.surfaceWaterDepthResults_layerid = ''
        if layer_id == self.surfaceWaterLevelResults_layerid:
            self.surfaceWaterLevelResults = None
            self.surfaceWaterLevelResults_layerid = ''

    def selectOutputFolder(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getExistingDirectory()
        if file_path:
            self.txtOutputFolder.setText(file_path)

    def openGroundModelRaster(self):

        if not self.groundModelQGISLayer is None:
            ret = QMessageBox.question(
                self, 'Warning', 'A ground model is already in use.  Are you sure you want to open a new one?', QMessageBox.Yes | QMessageBox.No)
            if ret == QMessageBox.Yes:
                self.thisProject.removeMapLayer(self.groundModelQGISLayer)
                # QgsProject.instance().removeMapLayer(self.groundModelQGISLayer)
                self.groundModelQGISLayer_layerid = ''
                self.groundModelQGISLayer = None
                self.groundModelRMG = None
                self.groundModelElevationData = None
                self.groundModelRMG_Filled = None
                self.groundModelElevationData_Filled = None
                self.txtGroundModelQGISLayer.setText("")
            else:
                return

        # Create a file dialog
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("ASC files (*.asc)")
        file_path, _ = file_dialog.getOpenFileName()
        if file_path:

            aLayer = QgsRasterLayer(file_path, 'LandLabs Ground Model')
            self.groundModelQGISLayer = self.thisProject.addMapLayer(aLayer)
            self.groundModelQGISLayer_layerid = self.groundModelQGISLayer.id()
            (self.groundModelRMG, self.groundModelElevationData) = read_esri_ascii(
                file_path, name='LandLabs Ground Model')
            (self.groundModelRMG_Filled, self.groundModelElevationData_Filled) = read_esri_ascii(
                file_path, name='LandLabs Ground Model Filled')
            self.txtGroundModelQGISLayer.setText(file_path)

    def onLayerModified(self):
        self.updateSimulationStats()

    def updateSimulationStats(self):
        self.hydrographDuration_s = self.dblSpnStormDuration.value() * (60*60)
        self.timeToPeak_s = self.hydrographDuration_s / 5
        if not self.pointSourceVectorLayer is None:
            self.maxNodeVol_m3 = self.pointSourceVectorLayer.maximumValue(
                self.pointSourceVectorLayer.fields().indexFromName('vol_m3'))
            if self.maxNodeVol_m3 > 0:
                self.maxNodePeakFlow_m3s = 0
                self.maxNodeAvgFlow_m3s = 0

            self.additionalLag_s = float(self.txtAdditionalLag.text())
            self.simulationDuration_s = self.hydrographDuration_s + self.additionalLag_s
            if not self.useVariableTimestep:
                self.fixedTimestep_s = self.spnMaxTimestep.value()
            self.animationTimestep_s = self.spnAnimationTimestep.value()
            self.totalInflowVolume_m3, isValid = self.pointSourceVectorLayer.aggregate(
                QgsAggregateCalculator.Aggregate.Sum, 'vol_m3')
            if not self.currentlySimming:
                self.currentInflowVolume_m3 = 0
            self.maxNodePeakFlow_m3s = self.getPeakFlow(self.maxNodeVol_m3)
            self.maxNodeAvgFlow_m3s = self.maxNodeVol_m3/self.hydrographDuration_s
        else:
            self.maxNodeVol_m3 = 0
            self.maxNodePeakFlow_m3s = 0
            self.maxNodeAvgFlow_m3s = 0
            # self.longestHydrographTime_s = 0
            self.additionalLag_s = 0
            self.simulationDuration_s = self.hydrographDuration_s + self.additionalLag_s
            self.fixedTimestep_s = 60
            self.animationTimestep_s = 300
            self.totalInflowVolume_m3 = 0
            self.currentInflowVolume_m3 = 0
            self.maxNodePeakFlow_m3s = 0
            self.maxNodeAvgFlow_m3s = 0
        self.updateDockWidget()

    def updateDockWidget(self):
        self.dblSpnStormDuration.setValue(self.hydrographDuration_s / (60*60))
        self.txtMaxNodeVol.setText(str(self.maxNodeVol_m3))
        self.txtAdditionalLag.setText(str(self.additionalLag_s))
        self.txtSimulationDuration.setText(str(int(self.simulationDuration_s)))
        if not self.useVariableTimestep:
            self.spnMaxTimestep.setValue(self.fixedTimestep_s)
        else:
            if not self.currentlySimming:
                self.spnMaxTimestep.setValue(0)
        self.txtTotalInflowVol.setText(
            str(round(self.totalInflowVolume_m3, 1)))
        self.txtCurrentInflowVol.setText(
            str(round(self.currentInflowVolume_m3, 1)))
        self.txtMaxNodePeakFlow.setText(
            str(round(self.maxNodePeakFlow_m3s, 4)))
        self.txtMaxNodeAvgFlow.setText(str(round(self.maxNodeAvgFlow_m3s, 4)))

    def createPointSourceVectorLayer(self):

        if not self.pointSourceVectorLayer is None:
            ret = QMessageBox.question(
                self, 'Warning', 'A file is already in use.  Are you sure you want to create a new one?', QMessageBox.Yes | QMessageBox.No)
            if ret == QMessageBox.Yes:
                self.thisProject.removeMapLayer(self.pointSourceVectorLayer)
                # QgsProject.instance().removeMapLayer(self.pointSourceVectorLayer)
                self.pointSourceVectorLayer = None
                self.txtPointSourceQGISLayer.setText("")
            else:
                return

        # Create the memory layer
        self.pointSourceVectorLayer = QgsVectorLayer(
            "Point", "LandLabs Point Source Layer", "memory", crs=self.thisProject.crs())
        # self.pointSourceVectorLayer = QgsVectorLayer(
        #     "Point", "LandLabs Point Source Layer", "memory", crs=QgsProject.instance().crs())
        self.pointSourceVectorLayer_layerid = self.pointSourceVectorLayer.id()

        # Set the fields for the memory layer
        pr = self.pointSourceVectorLayer.dataProvider()
        pr.addAttributes([QgsField("node_id", QVariant.String, 'integer', 25), QgsField(
            "vol_m3", QVariant.Double, 'double', 20, 6)])
        self.pointSourceVectorLayer.updateFields()

        self.thisProject.addMapLayer(self.pointSourceVectorLayer)
        # QgsProject.instance().addMapLayer(self.pointSourceVectorLayer)
        self.txtPointSourceQGISLayer.setText("QGIS Memory Vector Layer")

        self.pointSourceVectorLayer.layerModified.connect(self.onLayerModified)

        pr = self.pointSourceVectorLayer.dataProvider()
        self.pointSourceVectorLayer.startEditing()
        feature = QgsFeature()
        feature.setAttributes(["1", 30])
        point = QgsPointXY(435419, 336241)
        feature.setGeometry(QgsGeometry.fromPointXY(point))
        pr.addFeatures([feature])
        self.pointSourceVectorLayer.commitChanges()

    def createLowResGrid(self):

        lowResCellSize = 50  # use 10m grid
        if not self.groundModelRMG is None:
            self.groundModelRMG.at_node['topographic__elevation'] = self.groundModelElevationData

            original_shape = self.groundModelRMG.shape
            original_spacing = (self.groundModelRMG.dx, self.groundModelRMG.dy)
            original_xy_of_lower_left = self.groundModelRMG.xy_of_lower_left

            # create a new grid with a coarser resolution
            new_shape = (int((original_shape[0] * original_spacing[1]) // lowResCellSize),
                         int((original_shape[1] * original_spacing[0]) // lowResCellSize))
            new_spacing = (lowResCellSize, lowResCellSize)

            zValuesCopy = self.groundModelElevationData.copy()
            zValuesCopy_2D = zValuesCopy.reshape(original_shape)
            zValuesCopy_2D_Resized = cv.resize(
                zValuesCopy_2D, (new_shape[1], new_shape[0]), interpolation=cv.INTER_AREA)
            self.groundModelElevationData_LowRes = zValuesCopy_2D_Resized.ravel()

            self.groundModelRMG_LowRes = RasterModelGrid(
                new_shape, xy_spacing=new_spacing, xy_of_lower_left=original_xy_of_lower_left)
            self.groundModelRMG_LowRes.at_node['topographic__elevation'] = self.groundModelElevationData_LowRes

    def runOverlandFlowAnalysis(self):

        # Check if the raster layer is valid
        if not self.groundModelQGISLayer.isValid():
            QMessageBox.warning(
                self, 'Warning', 'Ground Model layer is not valid', QMessageBox.StandardButton.Ok)
            return

        if self.exportNetCDF:
            nc_file = f'{self.txtOutputFolder.text()}/temporalResults.nc'
            if os.path.exists(nc_file):
                QMessageBox.warning(
                    self, 'Warning', 'NetCDF file already exists', QMessageBox.StandardButton.Ok)
                return

        maxDepth_file = f'{self.txtOutputFolder.text()}/tempMaxDepth.asc'
        if os.path.exists(maxDepth_file):
            QMessageBox.warning(
                self, 'Warning', 'Max Depth file already exists', QMessageBox.StandardButton.Ok)
            return

        maxLevel_file = f'{self.txtOutputFolder.text()}/tempMaxLevel.asc'
        if os.path.exists(maxLevel_file):
            QMessageBox.warning(
                self, 'Warning', 'Max Level file already exists', QMessageBox.StandardButton.Ok)
            return

        excelNode_file = f'{self.txtOutputFolder.text()}/nodes.xlsx'
        if os.path.exists(excelNode_file):
            QMessageBox.warning(
                self, 'Warning', 'Node Excel file already exists', QMessageBox.StandardButton.Ok)
            return

        try:

            self.currentlySimming = True
            self.enableButtons()
            self.progressBar.setValue(0)

            self.groundModelRMG.at_node['topographic__elevation'] = self.groundModelElevationData
            self.groundModelRMG.at_node['surface_water__depth'] = np.zeros(
                self.groundModelElevationData.shape)
            if not self.groundModelRMG.has_field('surface_water__maxdepth', at='node'):
                self.groundModelRMG.add_field('surface_water__maxdepth', np.full(
                    self.groundModelElevationData.shape, self.maxDepthThreshold), at="node")
            if not self.groundModelRMG.has_field('surface_water__maxlevel', at='node'):
                self.groundModelRMG.add_field(
                    'surface_water__maxlevel', self.groundModelElevationData, at="node")
            if not self.groundModelRMG.has_field('surface_water__discharge_u', at='node'):
                self.groundModelRMG.add_field('surface_water__discharge_u', np.zeros(
                    self.groundModelElevationData.shape), at="node")
            if not self.groundModelRMG.has_field('surface_water__discharge_v', at='node'):
                self.groundModelRMG.add_field('surface_water__discharge_v', np.zeros(
                    self.groundModelElevationData.shape), at="node")

            # self.estimateExtentsOfFlooding()

            of = OverlandFlow(self.groundModelRMG,
                              steep_slopes=self.steepSlopes)
            # of = OverlandFlow(self.groundModelRMG, steep_slopes=False)
            sim_start = datetime(2022, 12, 19, 14, 0)
            elapsed_time = 0.0
            elapsed_animation_time = 0.0

            self.nodeHydrographs = pd.DataFrame()

            if self.exportNetCDF:
                # Create a new NetCDF file and open it for writing
                netcdf = netCDF4.Dataset(nc_file, "w", format="NETCDF4")

                # Get the dimensions of the model grid
                nrows, ncols = self.groundModelRMG.number_of_node_rows, self.groundModelRMG.number_of_node_columns

                # Create dimensions in the NetCDF file
                calendar = 'standard'
                timedim = netcdf.createDimension(dimname='time', size=None)
                d_x = netcdf.createDimension("x", ncols)
                d_y = netcdf.createDimension("y", nrows)

                # Create variables in the NetCDF file
                time_unit_out = "hours since " + \
                    sim_start.strftime("%Y-%m-%d %H:%M:%S")
                v_time = netcdf.createVariable("time", "f8", ("time",))
                v_time.units = time_unit_out
                v_time.long_name = 'time'

                v_x = netcdf.createVariable("x", "f4", ("x",))
                v_x.units = 'm'
                v_x.axis = 'X'  # Optional
                v_x.standard_name = 'projection_x_coordinate'
                v_x.long_name = 'x-coordinate in projected coordinate system'

                v_y = netcdf.createVariable("y", "f4", ("y",))
                v_y.units = 'm'
                v_y.axis = 'Y'  # Optional
                v_y.standard_name = 'projection_y_coordinate'
                v_y.long_name = 'y-coordinate in projected coordinate system'

                surface_water_depth = netcdf.createVariable(
                    "surface_water_depth", "f4", ("time", "y", "x",))
                surface_water_depth.units = 'm'
                surface_water_depth.long_name = 'Surface Water Depth'

                u_swd = netcdf.createVariable(
                    "u-swd", "f4", ("time", "y", "x",))
                u_swd.units = 'm**3 s**-1'
                u_swd.long_name = 'Surface Water Discharge u-component'

                v_swd = netcdf.createVariable(
                    "v-swd", "f4", ("time", "y", "x",))
                v_swd.units = 'm**3 s**-1'
                v_swd.long_name = 'Surface Water Discharge v-component'

                v_x[:] = np.unique(self.groundModelRMG.node_x)
                v_y[:] = np.unique(self.groundModelRMG.node_y)

            date_format = "%Y-%m-%d %H:%M:%S"
            # Run the model
            i = 0
            while elapsed_time < self.simulationDuration_s:

                if self.exportNetCDF:
                    if elapsed_time % self.animationTimestep_s == 0:
                        # Calculate the current timestep in seconds after 1970-1-1
                        current_time = (
                            sim_start + elapsed_time * timedelta(seconds=1))

                        v_time[i] = netCDF4.date2num(
                            current_time, time_unit_out)

                        surface_water_depth[i, :, :] = self.groundModelRMG.at_node["surface_water__depth"].reshape(
                            (nrows, ncols))

                        if i > 0:

                            self.groundModelRMG.at_node['surface_water__discharge_u'] = map_mean_of_horizontal_links_to_node(
                                self.groundModelRMG, 'surface_water__discharge')
                            self.groundModelRMG.at_node['surface_water__discharge_v'] = map_mean_of_vertical_links_to_node(
                                self.groundModelRMG, 'surface_water__discharge')

                        u_swd[i, :, :] = self.groundModelRMG.at_node['surface_water__discharge_u'].reshape(
                            (nrows, ncols))
                        v_swd[i, :, :] = self.groundModelRMG.at_node['surface_water__discharge_v'].reshape(
                            (nrows, ncols))

                        i += 1

                if not self.useVariableTimestep:
                    dt = self.fixedTimestep_s
                else:
                    dt = of.calc_time_step()
                    self.spnMaxTimestep.setEnabled(True)
                    self.spnMaxTimestep.setValue(dt)
                    self.spnMaxTimestep.setEnabled(False)

                if (elapsed_time + dt) >= (elapsed_animation_time + self.animationTimestep_s):
                    dt = (elapsed_animation_time +
                          self.animationTimestep_s) - elapsed_time
                    elapsed_animation_time = elapsed_time + dt

                dataDict = {'Time': elapsed_time}
                for feature in self.pointSourceVectorLayer.getFeatures():
                    attributes = feature.attributes()
                    geometry = feature.geometry()
                    px, py = geometry.asPoint()

                    vol_m3 = attributes[self.pointSourceVectorLayer.fields(
                    ).indexFromName('vol_m3')]

                    node_id = self.groundModelRMG.find_nearest_node(
                        (px, py))

                    newDepth, addedVolume, avgFlow = self.getNewDepthAtTimestep(
                        elapsed_time, dt, self.groundModelRMG.at_node['surface_water__depth'][node_id], vol_m3, self.groundModelRMG.spacing)

                    self.groundModelRMG.at_node['surface_water__depth'][node_id] = newDepth

                    self.currentInflowVolume_m3 += addedVolume
                    self.txtCurrentInflowVol.setText(
                        str(round(self.currentInflowVolume_m3, 1)))

                    dataDict[attributes[self.pointSourceVectorLayer.fields(
                    ).indexFromName('node_id')]] = avgFlow

                # self.nodeHydrographs = self.nodeHydrographs.append(
                #     dataDict, ignore_index=True)
                self.nodeHydrographs = pd.concat(
                    [self.nodeHydrographs, pd.DataFrame([dataDict])], ignore_index=True)

                # Run one time step of the model
                of.run_one_step(dt)

                # Compare the values in array1 and array2 and replace the values in array2 with the result
                self.groundModelRMG.at_node['surface_water__maxdepth'][:] = np.where(
                    self.groundModelRMG.at_node['surface_water__depth'] > self.groundModelRMG.at_node['surface_water__maxdepth'], self.groundModelRMG.at_node['surface_water__depth'], self.groundModelRMG.at_node['surface_water__maxdepth'])

                self.progressBar.setValue(
                    int((elapsed_time / self.simulationDuration_s) * 100))
                QApplication.instance().processEvents()

                # Update the elapsed time
                elapsed_time += dt

            self.groundModelRMG.at_node['surface_water__maxlevel'][:] = np.add(
                self.groundModelRMG.at_node['surface_water__maxdepth'], self.groundModelElevationData)

            if self.exportNetCDF:
                # Close the NetCDF file
                netcdf.close()

            if not self.surfaceWaterDepthResults is None:
                self.thisProject.removeMapLayer(self.surfaceWaterDepthResults)
                # QgsProject.instance().removeMapLayer(self.surfaceWaterDepthResults)
                self.surfaceWaterDepthResults = None
            if not self.surfaceWaterLevelResults is None:
                self.thisProject.removeMapLayer(self.surfaceWaterLevelResults)
                # QgsProject.instance().removeMapLayer(self.surfaceWaterLevelResults)
                self.surfaceWaterLevelResults = None

            write_esri_ascii(maxDepth_file, self.groundModelRMG,
                             'surface_water__maxdepth')
            write_esri_ascii(maxLevel_file, self.groundModelRMG,
                             'surface_water__maxlevel')

            # Load the ASCII raster into a raster layer object
            self.surfaceWaterDepthResults = QgsRasterLayer(
                maxDepth_file, 'Surface Water Max Depth Results Raster Layer')
            self.surfaceWaterDepthResults_layerid = self.surfaceWaterDepthResults.id()
            self.thisProject.addMapLayer(self.surfaceWaterDepthResults)
            # QgsProject.instance().addMapLayer(self.surfaceWaterDepthResults)
            self.pseudocolor_styling(
                self.surfaceWaterDepthResults, 'Blues', 10, True)

            self.surfaceWaterLevelResults = QgsRasterLayer(
                maxLevel_file, 'Surface Water Max Level Results Raster Layer')
            self.surfaceWaterLevelResults_layerid = self.surfaceWaterLevelResults.id()
            self.thisProject.addMapLayer(self.surfaceWaterLevelResults)
            # QgsProject.instance().addMapLayer(self.surfaceWaterLevelResults)
            self.pseudocolor_styling(
                self.surfaceWaterLevelResults, 'Blues', 10, True)

            if self.exportNetCDF:
                surfaceWaterTemporalResults = QgsMeshLayer(
                    nc_file, "surface_water_depth", "mdal")
                self.thisProject.addMapLayer(surfaceWaterTemporalResults)
                # QgsProject.instance().addMapLayer(surfaceWaterTemporalResults)

            self.nodeHydrographs.to_excel(excelNode_file)
            os.system('start "excel" ' + f'"{excelNode_file}"')

            self.currentlySimming = False
            self.enableButtons()
            self.progressBar.setValue(0)

        except:
            QMessageBox.warning(
                self, 'Exception', traceback.format_exc(), QMessageBox.Ok)
            if self.exportNetCDF:
                if not netcdf is None:
                    netcdf.close()

    def getNewDepthAtTimestep(self, currentTime: float, timestep: float, currentDepth: float, totalVolume_m3: float, cellSizes: float):

        if self.useUH:
            Qp = self.getPeakFlow(totalVolume_m3)

            elapsedTime = currentTime + timestep

            if elapsedTime <= 1.25 * self.timeToPeak_s:
                startFlow = (Qp / 2) * (1 - math.cos(math.pi *
                                                     currentTime / self.timeToPeak_s))
                endFlow = (Qp / 2) * (1 - math.cos(math.pi *
                                                   elapsedTime / self.timeToPeak_s))
                avgFlow = (startFlow + endFlow) / 2
            else:
                startFlow = 4.34 * Qp * \
                    math.exp(-1.3*currentTime/self.timeToPeak_s)
                endFlow = 4.34 * Qp * \
                    math.exp(-1.3*elapsedTime/self.timeToPeak_s)
                avgFlow = (startFlow + endFlow) / 2
        else:
            Qp = totalVolume_m3 / self.hydrographDuration_s
            avgFlow = Qp

        dx, dy = cellSizes
        newVol = avgFlow * timestep
        deltaDepth = newVol / (dx * dy)
        return (currentDepth + deltaDepth, newVol, avgFlow)

    def getPeakFlow(self, vol_m3: float):
        if self.useUH:
            return vol_m3 / (self.timeToPeak_s * 1.39)
        else:
            return vol_m3 / self.hydrographDuration_s

    def color_ramp_items(self, colormap: str, minimum: float, maximum: float, nclass: int) -> List[QgsColorRampShader.ColorRampItem]:
        delta = maximum - minimum
        fractional_steps = [i / nclass for i in range(nclass + 1)]
        ramp = QgsStyle().defaultStyle().colorRamp(colormap)
        colors = [ramp.color(f) for f in fractional_steps]
        steps = [minimum + f * delta for f in fractional_steps]
        return [
            QgsColorRampShader.ColorRampItem(step, color, str(step))
            for step, color in zip(steps, colors)
        ]

    def pseudocolor_styling(self, layer, colormap: str, nclass: int, makeFirstItemTransparent: bool = False) -> None:
        stats = layer.dataProvider().bandStatistics(1, QgsRasterBandStats.All)
        minimum = stats.minimumValue
        maximum = stats.maximumValue

        ramp_items = self.color_ramp_items(colormap, minimum, maximum, nclass)

        if makeFirstItemTransparent:
            myColor = ramp_items[0].color
            myColor.setAlpha(0)
            myItem = QgsColorRampShader.ColorRampItem(
                ramp_items[0].value, myColor, ramp_items[0].label)
            ramp_items[0] = myItem

        # shader_function = QgsColorRampShader()
        shader_function = QgsColorRampShader(minimum, maximum)
        shader_function.setClassificationMode(QgsColorRampShader.EqualInterval)
        shader_function.setColorRampItemList(ramp_items)

        raster_shader = QgsRasterShader()
        raster_shader.setRasterShaderFunction(shader_function)

        renderer = QgsSingleBandPseudoColorRenderer(
            layer.dataProvider(), 1, raster_shader)
        layer.setRenderer(renderer)
        layer.triggerRepaint()
