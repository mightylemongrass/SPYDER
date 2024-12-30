# Solar Prominence Yolo DEtectoR (SPYDER)

This code offers a autonomously method to detect solar prominences from solar images through machine learning and computer vision techniques, specifically the Yolov6 algorithm. The program outputs the coordinates of the prominences written in polar coordinates into a csv file. It is capable of batch-running through parameters inputed from a .yaml file, as well as running through individual files via a UI. The code was written in python, with the PYQT library used to create the user interface. 

![Screenshot 2024-10-30 194332](https://github.com/user-attachments/assets/d0e07f2b-dda2-428a-a588-63de99c03eef)

## Prerequisites for Running Program

Python 3  
Pyqt 5 or higher  
Astropy  
OpenCV  4 or higher  

## Running the Program

### Using the UI

```
python main-v4.py
```

Simply run the program without any parameters. Afterwards, the UI will load in. To select an images to run through the program, either click the "Open Fits Folder" button and choose the directory containing the fits files of the solar images or enter the repository into the search bar and click enter. (IMAGES MUST BE FITS FILES) This should cause the bar on the right to be filled in with all fits files contained within the directory. To select an image, simply select on the name of the file and click "select file". The image should then appear on the screen. 

### Batch Running through Multiple Files

If the following command is run, then the UI will not be used and the algorithm will be run through all fits files within a selected folder. 

```
python main-v4.py --cmd-mode True --yaml-path "/a.yaml"
```
The yaml path will contain the following parameters and must be in this format (example yaml file is in the repo):  

<img width="355" alt="Screenshot 2024-12-30 at 3 44 06 PM" src="https://github.com/user-attachments/assets/2f8a532d-be1f-4f4b-9427-e6d38be9d3ac" />

This yaml file will cause the program to process every fits file located in the input folder and store the bounding box data as well as the images (if save_images is set to True) in the output folder.   
The bounding box data will be in this format:

<img width="680" alt="Screenshot 2024-12-30 at 3 49 45 PM" src="https://github.com/user-attachments/assets/8f4c942f-bfc5-405d-be4c-90bc2eb87149" />

An alternative way to run the program can be done using the following command: 

```
python main-v4.py --cmd-mode True --yaml-path /a.yaml --input-path /inputfolder --output-path /outputfolder"
```

This essentially executes the same code as shown above, except overriding the input and output paths shown in the yaml. 

## Authors
Ian Kim  
Vasyl Yurchyshyn  

## Version

Version 0.7 (beta)

## License

SPYDER is distributed under the terms of the MIT license. All new contributions must be made under this license.  
