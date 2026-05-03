


#################################################################################################
#
#  Main program file for SPYDER PROGRAM
#  Consists of gathering input for UI and batch-running as well as code for batch-running
#  Uses functions from utils.py and also ui.py if ui is used
#
#################################################################################################


from utils import *
from astropy.io import fits
import copy
import glob
import argparse
import yaml
import sys
import ultralytics
sys.stdout.flush()

def infer(big_image, conf_thres, overlap_thres, usrdevice, usemiddle=False):
        duplicate = copy.deepcopy(big_image)
        size = 640

        H = big_image.shape[0]
        W = big_image.shape[1]

        coordinates = []
        ind1 = 0
        ind2 = 0
        if H < size:
            big_image = padding(big_image, 0, size)
        if W < size:
            big_image = padding(big_image, 1, size)

        if usemiddle:
            width = []
            height = []
            while True:
                if ind1+size >= W:
                    width.append(W-size)
                    break
                else:
                    width.append(ind1)
                    ind1 += int(size*3/4)
            while True:
                if ind2+size > H:
                    height.append(H-size)
                    break                    
                else:
                    height.append(ind2)
                    ind2 += int(size*3/4)
            for x in width:
                coordinates.append([x, height[0]])
                coordinates.append([x, height[-1]])
            for y in height[1:-1]:
                coordinates.append([width[0], y])
                coordinates.append([width[-1], y])
            

        else:
            while True:
                temp = []
                
                
                if ind1+size >= W:
                    temp.append(W-size)
                    ind1 = -1
                else:
                    temp.append(ind1)
                    ind1 += int(size/2)
                if ind2+size > H:
                    temp.append(H-size)
                    ind2 = H-size
                    
                else:
                    temp.append(ind2)
                    if ind1 == -1:
                        if ind2 == H-size:
                            coordinates.append(temp)
                            break
                        ind2 += int(size/2)
                        ind1 = 0


                coordinates.append(temp)


        overlap_checker = np.zeros([H, W], dtype=np.uint8)
        returned = []

        for ind, (x, y) in enumerate(coordinates):
            image = big_image[y:y+size, x:x+size]
            image = cv2.resize(image, (640, 640)) 

            results = model(image, device=usrdevice, verbose=False)

            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                num_filtered_detections = 0
            else:
                boxes = result.boxes
                class_ids = boxes.cls
                confs = boxes.conf

                target_class_id = 0

                mask = (class_ids == target_class_id) & (confs >= conf_thres)

                filtered_boxes = boxes[mask]

                result.boxes = filtered_boxes
                
                det = filtered_boxes.data.cpu()

                for *xyxy, conf, cls in reversed(det):

                    x1 = int(xyxy[0]) + x
                    y1 = int(xyxy[1]) + y
                    x2 = int(xyxy[2]) + x
                    y2 = int(xyxy[3]) + y

                    block = overlap_checker[y1:y2, x1:x2].astype(np.float32)
                    block_mean = np.mean(block)
                    
                    if block_mean > overlap_thres:
                        continue
                    
                    overlap_checker[y1:y2, x1:x2] = 255
                    new_xyxy = [x1, y1, x2, y2]
                    line = [int(cls), new_xyxy, float(conf)] 
                    returned.append(line)

                    cv2.rectangle(duplicate, (x1, y1), (x2, y2), (255, 0, 0), 3)
                

        overlap_checker = np.zeros([H, W], dtype=np.uint8)

        returned_boxes = []


        for n in returned[::-1]:
            
            x1, y1, x2, y2 = n[1]
            

            block = overlap_checker[y1:y2, x1:x2].astype(np.float32)
            block_mean = np.mean(block)
        
            if block_mean > overlap_thres:
                continue
            overlap_checker[y1:y2, x1:x2] = 255
            returned_boxes.append(n)

        return returned_boxes
    

    


def custom_folder(filepath, device, output_file_loc, yolo_dir, middle_option, conf_threshold=25, iou_threshold=0.45, save_images=False, start_time=None, end_time=None):
    format_sd = [int(n) for n in start_time.split('/')]
    format_ed = [int(n) for n in  end_time.split('/')]
    year_list = next(os.walk(filepath))[1]
    start_year = year_list.index(str(format_sd[2]))

    csv_saved_bboxes = []

    firstmon=True
    firstday=True
    quitloop=False
    for yr in year_list[start_year:]:
        month_list = next(os.walk(os.path.join(filepath, yr)))[1]
        start_month=0
        if firstmon:
            if format_sd[0] < 10:
                start_month = int(month_list.index('0'+str(format_sd[0])))
            else:
                start_month = int(month_list.index(str(format_sd[0])))
            firstmon = False
        for mon in month_list[start_month:]:
            day_list = next(os.walk(os.path.join(filepath, yr, mon)))[1]
            start_day=0
            if firstday:
                if format_sd[1] < 10:
                    start_day = int(day_list.index('0'+str(format_sd[1])))
                else:
                    start_day = int(day_list.index(str(format_sd[1])))
                firstday = False
            for day in day_list[start_day:]:
                if int(day) > format_ed[1] and int(yr)==format_ed[2] and int(mon)==format_ed[0]:
                    quitloop=True
                    break
                else:
                    file_loc = os.path.join(filepath, yr, mon, day, next(os.walk(os.path.join(filepath, yr, mon, day)))[1][0])
                    csv_saved_bboxes.extend(multifile(file_loc, device, output_file_loc, yolo_dir, middle_option, conf_threshold, iou_threshold, save_images=save_images, date=True))
            if int(mon) > format_ed[0] and int(yr)==format_ed[2]:
                quitloop=True
                break
            if quitloop==True:
                break
        
        if quitloop==True:
            break          
        if int(yr) > format_ed[2]:
            break
    #print(os.path.join(output_file_loc, start_time+"_to_"+end_time+"_bboxes.csv"))
    save_files(csv_saved_bboxes, os.path.join(output_file_loc, str(format_sd[2])+str(format_sd[0])+str(format_sd[1])+"_to_"+str(format_ed[2])+str(format_ed[0])+str(format_ed[1])+"_bboxes.csv"))
    


def multifile(filepath, device, output_file_loc, yolo_dir, middle_option, conf_threshold=25, iou_threshold=0.45, save_images=False, date=False):
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
    global model
    model = ultralytics.YOLO(yolo_dir)

    fits_list = glob.glob(os.path.join(filepath, "*.fts"))
    csv_save_file = []
    count = 0
    for fits_file in fits_list:
        small_file_path = os.path.basename(fits_file)
        print("processing data: " + small_file_path + " status: " + str(100* count / len(fits_list)) + "%")
        count += 1
        try:
            hdul = fits.open(fits_file)
            img = np.array(hdul[0].data)

            img = img/np.max(img)
            normal_image = to_rgb(img)

            dark, bright = get_stat(img)
            sun_image = np.clip(img-dark, 0, 1)
            sun_image = cv2.convertScaleAbs(to_rgb(sun_image), alpha=3, beta=50)

            sun_image_extra = cv2.cvtColor(sun_image, cv2.COLOR_BGR2GRAY)
            sun_image_extra = cv2.medianBlur(sun_image_extra,5)
            circles = cv2.HoughCircles(sun_image_extra, cv2.HOUGH_GRADIENT,2,1200)
            circle = circles[0,0,:]
            sun_cx = int(circle[0])
            sun_cy = int(circle[1])
            sun_radius = int(circle[2])

            normal_image = cv2.cvtColor(normal_image, cv2.COLOR_RGB2BGR)
            sun_image = cv2.cvtColor(sun_image, cv2.COLOR_RGB2BGR)
            sun_image2 = cv2.convertScaleAbs(normal_image, alpha=0.9, beta=-10)

            boxes = infer(sun_image, conf_threshold*0.01, iou_threshold, device, middle_option) 
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
        except Exception as error: 
            print(f"An exception occurred: {error}")
            pass
    if date==True:
        return csv_save_file
    else:
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
    parser.add_argument('--start-date', default=None, type=str, help='start date of photos (ONLY FOR NJIT)')
    parser.add_argument('--end-date', default=None, type=str, help='end date of photos (ONLY FOR NJIT)')
    return parser

def yaml_startup(yaml_filepath, input_file=None, output_file=None, start=None, end=None):
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
    
    if start != None and end != None:
        custom_folder(directory, device, output_file, yolo_dir, middle_option, conf_threshold, overlap_threshold, save_images=save_images, start_time=start, end_time=end)
    else:
        multifile(directory, device, output_file, yolo_dir, middle_option, conf_threshold, overlap_threshold, save_images=save_images)

args = args_parser().parse_args()



if args.input_path is not None and args.output_path is not None:
    if args.start_date is not None and args.end_date is not None:
        yaml_startup(args.yaml_path, args.input_path, args.output_path, start=args.start_date, end=args.end_date)
    else:
        yaml_startup(args.yaml_path, args.input_path, args.output_path)
else:
    if args.start_date is not None and args.end_date is not None:
        yaml_startup(args.yaml_path, start=args.start_date, end=args.end_date)
    else:
        yaml_startup(args.yaml_path)
