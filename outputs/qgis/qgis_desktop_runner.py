# Paste this code into QGIS Python Console (Ctrl+Alt+P):
from qgis.core import QgsProject
project_path = r"C:\Users\akshi\Downloads\geospatial_project\outputs\qgis\chhattisgarh_mrv_project.qgz"
QgsProject.instance().read(project_path)
print("Successfully loaded: Chhattisgarh Clipped FTW Map Project!")
