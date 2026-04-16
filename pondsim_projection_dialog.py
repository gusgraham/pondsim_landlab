from pondsim_helper import *

from ui_elements.Ui_pondsim_projection_dialog_base import Ui_Dialog


class pondsim_projectionDialog(QtWidgets.QDialog, Ui_Dialog):

    def __init__(self, parent=None):
        """Constructor."""
        super(pondsim_projectionDialog, self).__init__(parent)
        self.setupUi(self)

        self.btnOK.clicked.connect(self.onAccept)
        self.btnCancel.clicked.connect(self.onReject)

    def onAccept(self):
        self.accept()

    def onReject(self):
        self.reject()
