


#################################################################################################
#
#  Main program file for SPYDER PROGRAM
#  Consists of gathering input for UI and batch-running as well as code for batch-running
#  Uses functions from utils.py and also ui.py if ui is used
#
#################################################################################################




from ui import *
import argparse
import yaml

def multifile(filepath, device, output_file_loc, yolo_dir, middle_option, conf_threshold=0.05, iou_threshold=0.45, save_images=False):
    '''
    code for batch running algorithm through a folder of fits files
    Utilizes functions from utils.py, as well as the algorithm itself
    Function is called by yaml startup function (called whenever arguments are put into the command and yaml is inputed)
    
    @param filepath: filepath containing fits files
    @param device: type of device the algorithm will be run on (cpu or gpu number)
    @param output_file_loc: location where outputed data such as bounding boxes/images will be stored
    @param yolo_dir: directory of yolo checkpoint file
    @param conf_threshold: confidence setting for algorithm (higher confidence, more recall less precision)
    @param iou threshold: iou (intersection over union) or overlap settting for algorithm (higher threshold, more precision less recall)
    @param save_images: option to save images to output folder
    
    '''
    inferer = Inferer(yolo_dir, device, 640)
    fits_list = glob.glob(os.path.join(filepath, "*.fts"))
    csv_save_file = []
    for fits_file in fits_list:
        small_file_path = os.path.basename(fits_file)
        try:
            hdul = fits.open(fits_file)
            img = np.array(hdul[0].data)

            img = img/np.max(img)
            normal_image = to_rgb(img)

            sun_image = cv2.convertScaleAbs(to_rgb(img), alpha=3, beta=50)

            sun_image_extra = (img*255).astype(np.uint8)
            sun_image_extra = cv2.medianBlur(sun_image_extra,5)
            circles = cv2.HoughCircles(sun_image_extra, cv2.HOUGH_GRADIENT,2,1200)
            circle = circles[0,0,:]
            sun_cx = int(circle[0])
            sun_cy = int(circle[1])
            sun_radius = int(circle[2])

            normal_image = cv2.cvtColor(normal_image, cv2.COLOR_RGB2BGR)
            sun_image = cv2.cvtColor(sun_image, cv2.COLOR_RGB2BGR)
            sun_image2 = cv2.convertScaleAbs(normal_image, alpha=0.9, beta=-10)
            color_image = cv2.cvtColor(cv2.applyColorMap(sun_image2, cv2.COLORMAP_HOT), cv2.COLOR_RGB2BGR)
            boxes = inferer.inferv2(sun_image, conf_threshold, iou_threshold,\
                                                None, False, 1000, 0.25, 90, usemiddle=middle_option)
            polar_coords = []

            if save_images:
                duplicate = copy.deepcopy(sun_image)

            duplicate2 = copy.deepcopy(sun_image)
            cv2.circle(duplicate2, (sun_cx, sun_cy), 7, (0, 0, 255), -1)
            cv2.circle(duplicate2, (sun_cx, sun_cy), sun_radius, (255, 0, 255), 2)

            if save_images:
                cv2.circle(duplicate, (sun_cx, sun_cy), 7, (0, 0, 255), -1)
                cv2.circle(duplicate, (sun_cx, sun_cy), sun_radius, (255, 0, 255), 2)
                saved_bboxes, polar_coords, duplicate = format_boxes(boxes, sun_radius, sun_cx,\
                                    sun_cy, duplicate2, show_boxes=True, img_dir=small_file_path, duplicate=duplicate)

                base, ext = os.path.splitext(small_file_path)
                cv2.imwrite(os.path.join(output_file_loc, base+".png"), duplicate)
            else:
                saved_bboxes, polar_coords, duplicate = format_boxes(boxes, sun_radius, sun_cx,\
                                    sun_cy, duplicate2, ui=True, show_boxes=False, img_dir=small_file_path)
            csv_save_file.extend(saved_bboxes)
        except:
            pass
    print(os.path.join(output_file_loc, "saved_bboxes.csv"))
    save_files(csv_save_file, os.path.join(output_file_loc, "saved_bboxes.csv"))
        

def args_parser():
    '''
    returns arguments from terminal
    '''
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Training', add_help=True)
    parser.add_argument("--cmd-mode", action="store_true", help="Enable yaml/command line mode")
    parser.add_argument('--yaml-path', default='parameters.yaml', type=str, help='path of yaml (--use-yaml argument must be set to True)')
    parser.add_argument('--input-path', default=None, type=str, help='fits file locations only used when yaml is not used')
    parser.add_argument('--output-path', default=None, type=str, help='output file locations when yaml is not used')
    return parser

def yaml_startup(yaml_filepath, input_file=None, output_file=None):
    '''
    reads in yaml settings and calls main function with these settings
    
    @param input_file: optional input argument overriding yaml
    @param output_file: optional output argument overriding yaml
    '''
    with open(yaml_filepath, 'r') as file:
        data = yaml.safe_load(file)

    device = data['device']
    conf_threshold = data['yolo_conf_threshold']
    overlap_threshold = data['overlap_threshold']
    yolo_dir = data['yolo_dir']
    save_images = data['save_images']
    middle_option = data['better_optimization']

    if input_file == None and output_file == None:
        directory = data['file_path_of_data']
        output_file = data['output_folder_location']
    else:
        directory = input_file
        output_file = output_file
    multifile(directory, device, output_file, yolo_dir, middle_option, conf_threshold, overlap_threshold, save_images=save_images)

args = args_parser().parse_args()


print(args)

if not args.cmd_mode: # Create the application and the main window
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())

else: # Uses yaml file as input instead
    if args.input_path is not None and args.output_path is not None:
        yaml_startup(args.yaml_path, args.input_path, args.output_path)
    else:
        yaml_startup(args.yaml_path)