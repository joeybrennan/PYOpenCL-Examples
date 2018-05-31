import generateDataopenCL as genData
import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


from PyQt5 import uic, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

qtCreatorFile = "doubleslitui_surf.ui" # Enter file here.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class Main(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.plotField.clicked.connect(self.plotfield)

        
        #Notice that an empty figure instance was added to our plotting window in the initialization method. 
        #This is necessary because our first call to changefig will try to remove a previously displayed figure,
        #which will throw an error if one is not displayed. 
        #The empty figure serves as a placeholder so the changefig method functions properly.
        fig = Figure()
        fig2 = Figure()
        self.addmpl(fig)
        self.addmpl_2(fig2)

     
    
    def plotfield(self):
        wavelen = float(self.wavelen.toPlainText())
        modenum = int(self.modeNum.toPlainText())
        Zin = float(self.z.toPlainText())
        
        AN = genData.generateAN(wavelen, modenum)
        
        ET,ETz,x,z =genData.generateET(wavelen, modenum, AN, Zin)
        
        figx = Figure()
        ax1f1 = figx.add_subplot(111)
        ax1f1.set_title('Beam Pattern')

        ax1f1.plot(x,np.abs(ETz)**2,label='Beam Pattern at %s m' %Zin)
        ax1f1.set_xlabel('x [m]')
        ax1f1.set_ylabel('Intensity')
        ax1f1.legend(loc='best')
        ax1f1.grid()
        
        self.rmmpl()
        self.addmpl(figx)
        
        figx = Figure()
        ax1f1 = figx.add_subplot(111)
 
        # Make data.
        X = x
        Y = z
        ET = np.asanyarray(ET)
        Z = np.abs(ET)#/numpy.amax(np.abs(ETz))
        X, Y = np.meshgrid(X, Y)
        # Plot the surface.
        ax1f1.pcolor(Y,X,Z**2)
        ax1f1.axis([z.min(), z.max(), x.min(), x.max()])

        self.rmmpl_2()
        self.addmpl_2(figx)
        

        

    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.mplvl_0.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, 
                self.mplwindow, coordinates=True)
        self.mplvl_0.addWidget(self.toolbar)
        
    def addmpl_2(self, fig):
        self.canvas1 = FigureCanvas(fig)
        self.mplvl_1.addWidget(self.canvas1)
        self.canvas1.draw()
        self.toolbar1 = NavigationToolbar(self.canvas1, 
                self.mplwindow_1, coordinates=True)
        self.mplvl_1.addWidget(self.toolbar1)
        
    def rmmpl(self):
        self.mplvl_0.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl_0.removeWidget(self.toolbar)
        self.toolbar.close()
        
    def rmmpl_2(self):
        self.mplvl_1.removeWidget(self.canvas1)
        self.canvas1.close()
        self.mplvl_1.removeWidget(self.toolbar1)
        self.toolbar1.close()
    

    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    window = Main()
    window.show()
    app.exec_()
    