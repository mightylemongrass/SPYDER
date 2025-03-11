


#################################################################################################
#
#  Contains all the code for UI
#  Includes all code necessary for running UI
#  Uses functions from utils.py
#
#################################################################################################



from utils import *
from yolov6.core.inferer_prominence import Inferer
from astropy.io import fits
import copy
import glob

class MainImage(QWidget): 
    '''
    this is the displayed image component that is shown on the screen of the UI
    image in the form of numpy array can be used to update this display image
    '''
    def __init__(self, main_app): 
        '''
        initializes object
        main ui window is inputed in
        no output
        '''
        super(MainImage, self).__init__()
        self.main_app = main_app
        self.image_pixmap = QPixmap(640, 640)
        self.image_pixmap.fill(Qt.white)
        self.image_scale = 1.0
        self.setMinimumSize(500, 400)
        self.show()

    def set_image(self, numpy=False, numpy_img=None):
        '''
        changing the image
        '''
        if numpy == True:
            numpy_img = QImage(numpy_img, numpy_img.shape[1],\
                                numpy_img.shape[0], numpy_img.shape[1] * 3,QImage.Format_RGB888)
            self.image_pixmap = QPixmap.fromImage(numpy_img)
        self.update()
        return self.image_pixmap.width(), self.image_pixmap.height()

    def paintEvent(self, event):
        '''
        changing the dimensions of the image when window is resized
        '''
        painter = QPainter()
        painter.begin(self)
        if self.image_pixmap and self.image_pixmap.size().width() > 0:
            paint_w = float(self.size().width())
            paint_h = float(self.size().height())
            image_w = float(self.image_pixmap.size().width())
            image_h = float(self.image_pixmap.size().height())
            
            resized_w = paint_w
            resized_h = paint_w * image_h / image_w

            if resized_h > paint_h:
                resized_w = paint_h * image_w / image_h
                resized_h = paint_h
            self.image_scale = resized_w / image_w
            resized = self.image_pixmap.scaled(int(resized_w), int(resized_h))
        painter.drawPixmap(0, 0, resized)
        painter.end()
            
    def mousePressEvent(self, mouse_event):
        pass

    def mouseMoveEvent(self, mouse_event):
        pass

    def mouseReleaseEvent(self, mouse_event):
        pass
        

class MainTool(QWidget):
    '''
    UI component of the toolbar located on the left side of the UI window
    includes all parts of the toolbar with all its widgets (i.e. buttons, sliders, etc.)
    when widgets are interacted with, returns data back
    '''
    
    def __init__(self, main_app):
        '''
        initalization of widgets on the toolrack
        '''
        super(MainTool, self).__init__()
        self.main_app = main_app
        self.resize(170, 200)
        self.intValidator = QIntValidator()

        self.detector = QPushButton('Detect Prominences')
        self.detector.clicked.connect(self.main_app.detector_button)
        
        self.original = QRadioButton("original image")
        self.original.setChecked(True)
        self.original.toggled.connect(main_app.btnstate_normal)

        self.modified_image = QRadioButton("modified image")
        self.modified_image.toggled.connect(main_app.btnstate_contrast)

        self.color_image = QRadioButton("color image")
        self.color_image.toggled.connect(main_app.btnstate_color)

        self.checkcirclecenter = QCheckBox("Show Circle and Center")
        self.checkcirclecenter.setChecked(True)
        self.checkcirclecenter.toggled.connect(main_app.circle_center)
        self.checkbox = QCheckBox("Show Bboxes")
        self.checkbox.setChecked(True)
        self.checkbox.toggled.connect(main_app.box)

        self.listWidget = QListWidget()
        self.file_select = QPushButton('Select File')
        self.file_select.clicked.connect(main_app.selector)

        self.slidelabel1 = QLabel('confidence threshold:25')
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(100)
        self.slider1.setValue(25)
        self.slider1.setTickPosition(QSlider.NoTicks)
        self.slider1.valueChanged.connect(main_app.valuechanged)

        self.group1 = QGroupBox('Select Directory')
        layout1 = QVBoxLayout(self)
        layout1.addWidget(self.listWidget)
        layout1.addWidget(self.file_select)
        self.group1.setLayout(layout1)

        self.group2 = QGroupBox('Image Options')
        layout2 = QVBoxLayout(self)
        layout2.addWidget(self.original)
        layout2.addWidget(self.modified_image)
        layout2.addWidget(self.color_image)
        layout2.addWidget(self.checkcirclecenter)
        layout2.addWidget(self.checkbox)
        self.group2.setLayout(layout2)
        
        
        self.group3 = QGroupBox('Detect Options')
        layout3 = QVBoxLayout(self)
        layout3.addWidget(self.slidelabel1)
        layout3.addWidget(self.slider1)
        layout3.addWidget(self.detector)
        self.group3.setLayout(layout3)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.group1)
        layout.addWidget(self.group2)
        layout.addWidget(self.group3)

        self.setLayout(layout)
        self.show()

            
class MainApp(QMainWindow): 
    '''
    Entirety of the UI
    components of the UI such as the MainImage and MainTool are integrated into here
    when UI is interacted with, the data is returned back to the code
    UI can be updated through MainApp
    '''

    def __init__(self):
        '''
        initalizes all variables
        '''
        super(MainApp, self).__init__()
        self.title = 'SPYDER'
        self.left = 20
        self.top = 20
        self.width = 900
        self.height = 570
        self.working_image_path = ''
        self.displayed = False
        self.use_regular = True
        self.use_other = False
        self.use_color = False
        self.show_center = True
        self.show_circle_center = True
        self.show_boxes = True

        # You must replace the filepath below with the actual filepath of the checkpoint file on your computer

        weights = "/Users/iankim/Documents/Programming/SPYDER/weightsv3/best_ckpt.pt"
        img_size = 640
        self.inferer = Inferer(weights, "0", img_size)
        self.conf_thres = 0.05
        self.iou_thres = 0.45
        self.max_det = 1000
        self.agnostic_nms = False
        self.conf_threshold = 25
        self.overlap_threshold = 90
        self.selected = False
        self.selected_box = -1
        self.saved_bboxes = []
        self.output_file_loc = ""
        self.init_window()

    def init_window(self):
        '''
        initializes widgets on the window including the toolbar and the image
        '''
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.painter = MainImage(self)
        self.installEventFilter(self.painter)

        self.toolbox = MainTool(self)
        self.toolbox.setMaximumWidth(250)

        self.widget1 = QWidget(self)
        self.top_layout = QHBoxLayout(self)
        self.open_button = QPushButton('Open Fits Folder')
        self.open_button.clicked.connect(self.on_open_button)
        self.file_path = QLineEdit(self)
        self.file_path.setObjectName("file path")
        self.file_path.returnPressed.connect(self.on_pushButtonOK_clicked)
        self.top_layout.addWidget(self.file_path)
        self.top_layout.addWidget(self.open_button)
        self.widget1.setLayout(self.top_layout)
        self.widget1.setMaximumHeight(100)

        self.table = QTableWidget(self)
        table_header = ['Image','Left Edge','Right Edge','Height','Confidence','Area']
        a = self.frameGeometry().width()//4
        self.table_col_width = [a, a, a, a, a, a]
        self.table_row_height = self.frameGeometry().height()//20
        self.table.setColumnCount(len(table_header))
        self.table.setRowCount(3)
        self.table.horizontalHeader().hide()
        self.table.verticalHeader().hide()
        self.table.setFont(QFont("arial", 9))
        for col in range(len(table_header)):
            self.table.setColumnWidth(col, self.table_col_width[col])
            self.table.setItem(0, col, QTableWidgetItem(table_header[col]))

        self.save_button = QPushButton('Save Image and Boxes')
        self.save_button.clicked.connect(self.save_csv)
        self.delete_button = QPushButton('Delete Box')
        self.delete_button.clicked.connect(self.delete_box)

        self.bottom_buttons = QWidget(self)
        self.button_layout = QVBoxLayout()
        self.button_layout.addWidget(self.delete_button)
        self.button_layout.addWidget(self.save_button)
        self.bottom_buttons.setLayout(self.button_layout)

        self.bottom_widget = QWidget(self)
        self.bottom_layout = QHBoxLayout()
        self.bottom_layout.addWidget(self.table)
        self.bottom_layout.addWidget(self.bottom_buttons)
        self.bottom_widget.setLayout(self.bottom_layout)
        self.bottom_widget.setMaximumHeight(self.frameGeometry().height()//3)

        self.widget2 = QWidget(self)
        self.main_dock = QVBoxLayout()
        self.main_dock.addWidget(self.widget1)
        self.main_dock.addWidget(self.painter)
        self.main_dock.addWidget(self.bottom_widget)
        self.widget2.setLayout(self.main_dock)
        
        
        mainwidget = QWidget(self)
        layout = QHBoxLayout()
        layout.addWidget(self.toolbox)
        layout.addWidget(self.widget2)
        mainwidget.setLayout(layout)
        self.setCentralWidget(mainwidget)

    def close_app(self):
        '''
        closing window
        '''
        sys.exit()

    def on_pushButtonOK_clicked(self):
        '''
        changes image path when button is clicked
        '''
        self.working_image_path = self.file_path.text()
        self.file_path.setText(str(self.working_image_path))
        self.displayed = False
        self.selected = False
        csv_list = glob.glob(os.path.join(self.working_image_path, "*.fts"))
        self.toolbox.listWidget.clear()
        for csv in csv_list:
            csv_fn = os.path.basename(csv)
            listWidgetItem = QListWidgetItem(csv_fn)
            self.toolbox.listWidget.addItem(listWidgetItem)

    def redraw(self, r=False):
        '''
        reloading image when checkboxes are clicked, different image preset is used, or confidence value is changed
        '''
        if self.displayed == True:
            if self.selected == True:
                if self.use_regular == True:
                    duplicate = copy.deepcopy(self.normal_image)
                if self.use_other == True:
                    duplicate = copy.deepcopy(self.sun_image)
                if self.use_color == True:
                    duplicate = copy.deepcopy(self.color_image)
                if self.show_circle_center == True:
                    cv2.circle(duplicate, (self.sun_cx, self.sun_cy), 7, (0, 0, 255), -1)
                    cv2.circle(duplicate, (self.sun_cx, self.sun_cy), self.sun_radius, (255, 0, 255), 2)
                if self.show_boxes == True:
                    self.draw_arcs(duplicate)
                w, h = self.painter.set_image(numpy=True, numpy_img=duplicate)
                if r:
                    return duplicate

    def load_table(self):
        '''
        loads in table (called whenever table is updated)
        '''
        table_header = ['Image','Left Edge','Right Edge','Height','Confidence', 'Area']
        self.table.setRowCount(len(self.saved_bboxes) + 1)
        for row in range(len(self.saved_bboxes)):
            self.table.setRowHeight(row, self.table_row_height)
            for col in range(len(table_header)):
                self.table.setItem(row+1, col, QTableWidgetItem(self.saved_bboxes[row][col]))

    def draw_rect(self, duplicate):
        '''
        draws rectangular bounding boxes (not used)
        '''
        for line in self.boxes:
            x1, y1, x2, y2 = line[1]
            if float(line[2])>self.conf_threshold*0.01:
                if int(line[0]) == 1:
                    cv2.rectangle(duplicate, (x1, y1), (x2, y2), (255, 0, 0), 2)
 
    def draw_arcs(self, duplicate):
        '''
        draws polar bounding boxes
        '''
        ind = 0
        for conf, px1, py1, px2, py2, px3, py3, px4, py4 ,degree1, degree2, extra, area in self.polar_coords:
            if conf > self.conf_threshold*0.01:
                if self.selected_box >= 0 and self.selected_box == ind:
                    draw_ellipse(degree1, degree2, self.sun_radius, self.sun_cx, self.sun_cy, extra, duplicate, color=True)
                    cv2.line(duplicate, (px1, py1), (px2, py2), (255, 255, 0), 5)
                    cv2.line(duplicate, (px3, py3), (px4, py4), (255, 255, 0), 5)
                else:
                    draw_ellipse(degree1, degree2, self.sun_radius, self.sun_cx, self.sun_cy, extra, duplicate)
                    cv2.line(duplicate, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.line(duplicate, (px3, py3), (px4, py4), (0, 255, 0), 2)
            ind += 1

    def save_csv(self):
        '''
        saves the bounding boxes in a csv file
        '''
        if self.displayed == True:
            if self.selected == True:
                base = os.path.splitext(self.img_dir)[0]
                save_files(self.saved_bboxes, os.path.join(self.output_file_loc, base+".csv"))
                cv2.imwrite(os.path.join(self.output_file_loc, base+".png"), self.redraw(r=True))

    def delete_box(self):
        '''
        deletes selected row of table when delete is pressed
        '''
        if self.selected_box >= 0:
            del self.polar_coords[self.selected_box]
            del self.saved_bboxes[self.selected_box]
        self.load_table()
        self.redraw()

    def btnstate_normal(self):
        '''
        updates with original form of image (ran when original image button is pressed)
        '''
        self.use_regular = True
        self.use_other = False
        self.use_color = False
        if self.displayed:
            if self.selected:
                self.redraw()
            else:
                w, h = self.painter.set_image(numpy=True, numpy_img=self.normal_image)

    def btnstate_contrast(self):
        '''
        updates with improved contrast form of image (ran when modified image button is pressed)
        '''
        self.use_regular = False
        self.use_other = True
        self.use_color = False
        if self.displayed:
            if self.selected:
                self.redraw()
            else:
                w, h = self.painter.set_image(numpy=True, numpy_img=self.sun_image)

    def btnstate_color(self):
        '''
        updates with color image (ran when color image button is pressed)
        '''
        self.use_regular = False
        self.use_other = False
        self.use_color = True
        if self.displayed:
            if self.selected:
                self.redraw()
            else:
                w, h = self.painter.set_image(numpy=True, numpy_img=self.color_image)

    def circle_center(self):
        '''
        draws in circle and center on the image
        '''
        if self.show_circle_center == True:
            self.show_circle_center = False
        else:
            self.show_circle_center = True
        self.redraw()

    def box(self):
        '''
        shows/hides boxes
        '''
        if self.show_boxes == True:
            self.show_boxes = False
        else:
            self.show_boxes = True
        self.redraw()

    def on_open_button(self):
        '''
        opens the selected file
        '''
        if len(self.working_image_path) == 0:
            image_dir = os.getcwd()
        else:
            image_dir = os.path.dirname(self.working_image_path)
        self.working_image_path = QFileDialog.getExistingDirectory(self, 'Open File', image_dir)
        self.file_path.setText(str(self.working_image_path))
        self.displayed = False
        self.selected = False

        csv_list = glob.glob(os.path.join(self.working_image_path, "*.fts"))
        self.toolbox.listWidget.clear()
        for csv in csv_list:
            csv_fn = os.path.basename(csv)
            listWidgetItem = QListWidgetItem(csv_fn)
            self.toolbox.listWidget.addItem(listWidgetItem)

    def selector(self):
        '''
        displays the selected image file
        '''
        self.boxes = []
        try:
            image_path = self.file_path.text()
            self.output_file_loc = image_path
            if os.path.isfile(os.path.join(image_path, self.toolbox.listWidget.selectedItems()[0].text())):
                hdul = fits.open(os.path.join(image_path, self.toolbox.listWidget.selectedItems()[0].text()))
                self.img_dir = self.toolbox.listWidget.selectedItems()[0].text()
                img = np.array(hdul[0].data)

                # 1. normalize img first for detection
                img = img/np.max(img)
                # 2. * 255, to RGB
                self.normal_image = to_rgb(img)

                dark, bright = get_stat(img)
                self.sun_image = np.clip(img-dark, 0, 1)
                self.sun_image = cv2.convertScaleAbs(to_rgb(self.sun_image), alpha=3, beta=50)

                sun_image_extra = cv2.cvtColor(self.sun_image, cv2.COLOR_BGR2GRAY)
                sun_image_extra = cv2.medianBlur(sun_image_extra,5)
                circles = cv2.HoughCircles(sun_image_extra, cv2.HOUGH_GRADIENT,2,1200)
                circle = circles[0,0,:]
                self.sun_cx = int(circle[0])
                self.sun_cy = int(circle[1])
                self.sun_radius = int(circle[2])

                self.normal_image = cv2.cvtColor(self.normal_image, cv2.COLOR_RGB2BGR)
                self.sun_image = cv2.cvtColor(self.sun_image, cv2.COLOR_RGB2BGR)
                sun_image2 = cv2.convertScaleAbs(self.normal_image, alpha=0.9, beta=-10)
                self.color_image = cv2.cvtColor(cv2.applyColorMap(sun_image2, cv2.COLORMAP_HOT), cv2.COLOR_RGB2BGR)

                if self.use_regular == True:
                    w, h = self.painter.set_image(numpy=True, numpy_img=self.normal_image)
                elif self.use_color == True:
                    w, h = self.painter.set_image(numpy=True, numpy_img=self.color_image)
                else:
                    w, h = self.painter.set_image(numpy=True, numpy_img=self.sun_image)
                self.displayed = True
        except:
            pass

    def detector_button(self):
        '''
        runs image through the model and annotates image
        '''
        if self.displayed == True:

            self.boxes = self.inferer.inferv2(self.sun_image, self.conf_thres, self.iou_thres,\
                                None, self.agnostic_nms, self.max_det, self.conf_threshold*0.01, self.overlap_threshold, usemiddle=True)
            self.polar_coords = []

            if self.use_regular == True:
                duplicate = copy.deepcopy(self.normal_image)

            if self.use_other == True:
                duplicate = copy.deepcopy(self.sun_image)

            if self.use_color == True:
                duplicate = copy.deepcopy(self.color_image)

            duplicate2 = copy.deepcopy(self.sun_image)
            cv2.circle(duplicate2, (self.sun_cx, self.sun_cy), 7, (0, 0, 255), -1)
            cv2.circle(duplicate2, (self.sun_cx, self.sun_cy), self.sun_radius, (255, 0, 255), 2)

            if self.show_circle_center == True:
                cv2.circle(duplicate, (self.sun_cx, self.sun_cy), 7, (0, 0, 255), -1)
                cv2.circle(duplicate, (self.sun_cx, self.sun_cy), self.sun_radius, (255, 0, 255), 2)
            self.saved_bboxes, self.polar_coords, duplicate = format_boxes(self.boxes, self.sun_radius, self.sun_cx,\
                                 self.sun_cy, duplicate2, show_boxes=self.show_boxes, img_dir=self.img_dir, duplicate=duplicate)


            w, h = self.painter.set_image(numpy=True, numpy_img=duplicate)

            self.load_table()
            self.selected = True

    def valuechanged(self):
        '''
        updates bboxes based on changing confidence value
        '''
        self.toolbox.slidelabel1.setText('confidence threshold: ' + str(self.toolbox.slider1.value()))
        self.conf_threshold = self.toolbox.slider1.value()

        self.redraw()
        
    def on_run_button(self):
        if len(self.working_image_path) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText('No image selected')
            msg.setWindowTitle('Error')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

    def on_view_mode(self, mode):
        print("hello")
