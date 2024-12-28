# SPYDER
Solar Prominence Yolov6 Detector  

This code offers a autonomously method to detect solar prominences from solar images through machine learning and computer vision techniques, specifically the Yolov6 algorithm. The program outputs the coordinates of the prominences written in polar coordinates into a csv file. It is capable of batch-running through parameters inputed from a .yaml file, as well as running through individual files via a UI. The code was written in python, with the PYQT library used to create the user interface. 

![Screenshot 2024-10-30 194332](https://github.com/user-attachments/assets/d0e07f2b-dda2-428a-a588-63de99c03eef)

## Prerequisites for Running Program

Pyqt 5 or higher  
Astropy  
OpenCV  4 or higher  

## Running the Program

### Using the UI

Simply run the program without any parameters. Afterwards, the UI will load in. To select an images to run through the program, either click the "Open Fits Folder" button and choose the directory containing the fits files of the solar images or enter the repository into the search bar and click enter. (IMAGES MUST BE FITS FILES) This should cause the bar on the right to be filled in with all fits files contained within the directory. To select an image, simply select on the name of the file and click "select file". The image should then appear on the screen. 

### Batch Running through Multiple Files



## Authors
Ian Kim  
Vasyl Yurchyshyn  
