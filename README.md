# Solar Prominence Yolo DEtectoR (SPYDER)

SPYDER is a deep-learning artificial intelligence algorithm that offers an autonomous method to detect solar prominences from solar images through machine learning and computer vision techniques, specifically the Yolov6 algorithm. The program outputs the coordinates of the prominences written in polar coordinates into a csv file. It is capable of batch-running through parameters inputed from a .yaml file, as well as running through individual files via a UI. The code was written in python, with the PYQT library used to create the user interface. 

![image](https://github.com/user-attachments/assets/76ef54cb-a821-4d98-a8ac-60fea6f14996)

## Prerequisites for Running Program

Python 3  
Pytorch  
OpenCV  4 or higher  
Pyqt 5 or higher  
Astropy  
Pandas  
  
Simply install the following prerequisites in a conda environment using the requirements.txt file.  
```
pip install -r requirements.txt  
```

## Running the Program

### Using the UI

```
python main.py
```

Simply run the program without any parameters. Afterwards, the UI will load in automatically. To select an image to run through the program, either click the "Open Fits Folder" button and choose the directory containing the fits files of the solar images or enter the repository into the search bar and click enter. (IMAGES MUST BE FITS FILES) This should cause the bar on the right to be filled in with all fits files contained within the directory. To select an image, simply select on the name of the file and click "select file". The image should then appear on the screen. From here, you may press the detect prominences button to run the algorithm through the image. (may process instantaneously to around a minute depending on the speed of your computer) The prominences should appear on the image, and the table should be filled with the coordinates of their bounding boxes. At this point, optionally, you may adjust the confidence slider and the other image options to your liking. Afterwards, you may select a row and click delete if you wish to delete a bounding box, and then click Save Images and Boxes in order to save the bounding boxes in a csv file. 

### Batch Running through Multiple Files

If the following command is run, then the UI will not be used and the algorithm will be run through all fits files within a selected folder. 

```
python main.py --cmd-mode True --yaml-path "/a.yaml"
```
The yaml path will contain the following parameters and must be in this format (example yaml file is in the repo):  

<img width="711" alt="Screenshot 2025-01-30 at 8 44 58 PM" src="https://github.com/user-attachments/assets/2a861d49-0fe7-4110-adaa-6389b1d42ce1" />

This yaml file will cause the program to process every fits file located in the input folder and store the bounding box data as well as the images (if save_images is set to True) in the output folder.  
Here are the various parameters present in the yaml file explained:  
  
device: The device used for the AI model (either cpu or number corresponding to gpu)  
yolo_conf_threshold: confidence threshold for boxes  
overlap_threshold: overlap threshold for boxes  
file_path_of_data: input file path for folder containing fits files  
output_folder_location: where the images as well as csv file will be saved  
yolo_dir: checkpoint file location for yolo algorithm  
save_images: whether or not to save images  
better_optimization: saves time with better optimization if set to True  
  
  
The bounding box data will be in this format:

<img width="542" alt="Screenshot 2025-03-02 151536" src="https://github.com/user-attachments/assets/51ece66f-5112-492b-afdf-411e7cea57ea" />  

An alternative way to run the program can be done using the following command: 

```
python main.py --cmd-mode --yaml-path a.yaml --input-path /inputfolder --output-path /outputfolder
```

This essentially executes the same code as shown above, except overriding the input and output paths shown in the yaml for quick changes.  

If using a year/month/day folder system, you can use the following command:  

```  
python main.py --cmd-mode --yaml-path a.yaml --input-path /inputfolder --output-path /outputfolder --start-date 5/17/2024 --end-date 6/4/202  
```  
NOTE: This is only suitable when using the folder format of year/month/day/filename.fits, where the program will through all files from the start to end dates.   

## Steps for Installation  
  
First, download all files in the repository and save them in a single folder.  
Then, download the checkpoint file. 

The checkpoint file can be downloaded at this site:  
https://drive.google.com/drive/folders/1wCbNUMoNF6BgzvsFxPiHMqucphtQI7Gt  

If using the UI, modify the weights variable as shown below. 
  
<img width="919" alt="Screenshot 2025-01-30 at 8 42 03 PM" src="https://github.com/user-attachments/assets/619fe56f-e843-4d11-a2cd-0b31ddb1aa99" />

If batch running, simply modify the values in the yaml file. (shown above)  

## Authors
Ian Kim  ik9davis@gmail.com  
Vasyl Yurchyshyn  vasyl.yurchyshyn@njit.edu  

## Version

Version 0.7 (beta)

## License

SPYDER is distributed under the terms of the MIT license. All new contributions must be made under this license.  
