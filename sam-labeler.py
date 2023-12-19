import cv2
import os
import xml.etree.ElementTree as ET
from fastsam import SAM

"""
Usage:
    Set image_folder, output_folder, labels variables below
    Keys:
    a:      Previous image
    d       Next image
    0-9     Label selection
    r/m     Change mode (auto/manual)
    esc/q   Exit
    +/-     Increase or decrease confidence for SAM auto mode

Requeriments:
    OpenCV
    Pytorch (gpu support optional)
    Ultralytics

Citation:
    Original Fast Sam implementation from these grest guys:
    Xu Zhao, Wenchao Ding, Yongqi An, Yinglong Du, Tao Yu, Min Li, Ming Tang, Jinqiao Wang
    Can be found in Arxiv:
    https://arxiv.org/abs/2306.12156
"""


image_folder = '/home/pablo/DS/orange cookies/Images'
output_folder = '/home/pablo/DS/orange cookies/Annotations'
labels = ['foo','OK', 'NOK']
device = 'cpu' # or 'cuda:0'

class ImageEditor:
    def __init__(self, image_folder, output_folder, labels):
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.labels = labels
        self.images = sorted(os.listdir(image_folder))
        self.current_image_index = 0
        self.rectangles = []
        self.load_rectangles()
        self.drawing = False

        self.colors=[
                    (255, 0, 0),      # Blue
                    (0, 255, 0),      # Green
                    (0, 0, 255),      # Red
                    (0, 255, 255),    # Yellow
                    (255, 0, 255),    # Magenta
                    (255, 255, 0),    # Cyan
                    (0, 165, 255),    # Orange
                    (238, 130, 238),  # Violet
                    (255, 192, 203),  # Pink
                    (50, 205, 50)     # Lime Green
                ]
        self.current_rectangle = None
        self.selected_color=0
        # self.device = 'cpu'
        # self.device = 'cuda:0'
        self.sam=SAM(device=device)
        self.mode='manual'

    def run(self):
        while True:
            image_path = os.path.join(self.image_folder, self.images[self.current_image_index])
            image = cv2.imread(image_path)
            for rect in self.rectangles:
                find_index = lambda lst, s: lst.index(s) if s in lst else -1
                indice = find_index(self.labels, rect[0])
                if indice >= 0:
                    color=self.colors[indice]
                    label=self.labels[indice]
                else:
                    color=(200,200,200)
                    label='unknown'
                cv2.putText(image, label, rect[1][0], cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1, cv2.LINE_AA)
                cv2.rectangle(image, rect[1][0], rect[1][1], color, 2)
            if self.current_rectangle != None:
                cv2.rectangle(image, self.current_rectangle[0], self.current_rectangle[1], self.colors[self.selected_color], 2)

            cv2.putText(image, self.images[self.current_image_index], (0,20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

            if self.selected_color < len(self.labels):
                selected_label = self.labels[self.selected_color]
            else:
                selected_label = 'unknown ' + str(self.selected_color)

            cv2.putText(image, 'Selected label: '+selected_label, (0,40), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors[self.selected_color], 1, cv2.LINE_AA)
            cv2.putText(image, 'Mode: '+ self.mode, (0,60), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors[self.selected_color], 1, cv2.LINE_AA)
            cv2.putText(image, 'Confidence: '+ f"{self.sam.confidence:.1f}", (0,80), cv2.FONT_HERSHEY_DUPLEX, 0.7, self.colors[self.selected_color], 1, cv2.LINE_AA)
            cv2.imshow('FastSam Image Labeler', image)
            key = cv2.waitKey(1)

            if key == ord('d'):  # fw
                if self.current_image_index < len(self.images)-1:
                    self.current_image_index = self.current_image_index + 1
                    self.rectangles = []
                    self.load_rectangles()
            elif key == ord('a'):  # bw
                if self.current_image_index > 0:
                    self.current_image_index = self.current_image_index - 1
                    self.rectangles = []
                    self.load_rectangles()
            elif key == ord('m') or key == ord('r'):  # change mode
                if self.mode=='manual':
                    self.mode='auto'
                else:
                    self.mode='manual'
            elif key == ord('+'):  # increase confidence
                if self.sam.confidence <= .9:
                    self.sam.confidence += .1
            elif key == ord('-'):  # decrease confidence
                if self.sam.confidence >= .2:
                    self.sam.confidence -= .1
            elif key == 27 or key == ord('q'):  # ESC
                break
            elif 48 <= key <= 57: 
                self.selected_color=key-48

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.mode=='manual':
            self.drawing = True
            self.current_rectangle = [(x, y), (x, y)]

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_rectangle[1] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.mode=='manual':
            self.drawing = False
            x1, y1 = self.current_rectangle[0]
            x2, y2 = self.current_rectangle[1]
            if (x2 - x1)**2 > 20 and (y2 - y1)**2 > 20:
                if self.selected_color < len(self.labels):
                    self.rectangles.append((self.labels[self.selected_color],self.current_rectangle))
                else:
                    self.rectangles.append(('unknwon',self.current_rectangle))
            
            self.current_rectangle = None
            self.save_rectangles()
        
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.drawing = False
            self.current_rectangle = None
            self.remove_rectangle(x, y)
            self.save_rectangles()
            self.load_rectangles()
        
        elif event == cv2.EVENT_LBUTTONDOWN and self.mode=='auto':
            box = self.sam.inference(os.path.join(self.image_folder, self.images[self.current_image_index]), x, y)
            if  box != None:
                [x0, y0, w, h]= box
                x1 = x0 + w
                y1 = y0 + h
                print(x0, y0, x1, y1)
                print(self.rectangles)
                if self.selected_color < len(self.labels):
                    self.rectangles.append((self.labels[self.selected_color],[(x0, y0), (x1, y1)]))
                else:
                    self.rectangles.append(('unknwon',[(x0, y0), (x1, y1)]))
                self.current_rectangle = None
                self.save_rectangles()
                    

    def point_in_rect(self, rect, x, y):
        min_x=min(rect[0][0],rect[1][0])
        max_x=max(rect[0][0],rect[1][0])
        min_y=min(rect[0][1],rect[1][1])
        max_y=max(rect[0][1],rect[1][1])
        if (min_x <= x <= max_x and min_y <= y <= max_y):
            return True
        else:
            return False

    def remove_rectangle(self, x, y):
        self.rectangles = [rect for rect in self.rectangles if not self.point_in_rect(rect[1], x, y)]

    def save_rectangles(self):
        image_name = self.images[self.current_image_index]
        image_path = os.path.join(self.image_folder, image_name)
        image = cv2.imread(image_path)
        height, width, depth = image.shape

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = os.path.basename(self.image_folder)
        ET.SubElement(root, "filename").text = image_name
        ET.SubElement(root, "path").text = image_path

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(depth)

        for rect in self.rectangles:
            xmin, ymin = rect[1][0]
            xmax, ymax = rect[1][1]
            label= rect[0]
            object = ET.SubElement(root, "object")
            ET.SubElement(object, "name").text = label  # o el nombre del objeto que corresponda
            bndbox = ET.SubElement(object, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)

        tree = ET.ElementTree(root)
        tree.write(os.path.join(self.output_folder, os.path.splitext(self.images[self.current_image_index])[0] + ".xml"))



    def load_rectangles(self):
        self.rectangles = []
        image_name = os.path.splitext(self.images[self.current_image_index])[0]
        xml_file = os.path.join(self.output_folder, image_name + ".xml")

        if os.path.exists(xml_file):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for object in root.findall('object'):
                bndbox = object.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                label = object.find('name').text
                self.rectangles.append((label,[(xmin, ymin), (xmax, ymax)]))

if __name__ == "__main__":
    isExist = os.path.exists(output_folder)
    if not isExist:
        os.makedirs(output_folder)
    editor = ImageEditor(image_folder, output_folder, labels)
    cv2.namedWindow('FastSam Image Labeler')
    cv2.setMouseCallback('FastSam Image Labeler', editor.mouse_callback)
    editor.run()
