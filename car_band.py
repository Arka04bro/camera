from lxml import etree
import cv2
from glob import glob
import re

CAR_BRANDS_PATH = 'CAR_BRANDS/'

classes = {'lada': 0, 'kia': 1, 'nissan': 2, 'volkswagen': 3, 'chevrolet': 4,
           'ford': 5, 'mitsubishi': 6, 'renault': 7, 'hyundai': 8, 'opel': 9}

def extract_txt(img_path, xml_path, txt_path):
    img = cv2.imread(img_path)
    W = img.shape[1]
    H = img.shape[0]
    tree = etree.parse(xml_path)

    root = tree.getroot()
    with open(txt_path, 'w') as f:
        for child in root:
            if child.tag == 'object':
                for object_child in child:
                    coords = []
                    if object_child.tag == 'name':
                        f.write(str(classes.get(object_child.text))+' ')
                    if object_child.tag == 'bndbox':
                        for bbox_child in object_child:
                            coords.append(int(bbox_child.text))
                        xmin = coords[0]
                        xmax = coords[2]
                        ymin = coords[1]
                        ymax = coords[3]

                        w = (xmax - xmin) / W
                        h = (ymax - ymin) / H
                        xc = (xmin + (xmax - xmin) / 2) / W
                        yc = (ymin + (ymax - ymin) / 2) / H

                        f.write(str(xc)+' '+str(yc)+' '+str(w)+' '+str(h)+'\n')


xmls = sorted(glob(CAR_BRANDS_PATH + '*.xml'))

for xml_path in xmls:
    img_path = re.findall('[A-Za-z0-9_/]+', xml_path)[0]+'.jpg'
    txt_path = re.findall('[A-Za-z0-9_/]+', img_path)[0]+'.txt'
    extract_txt(img_path, xml_path, txt_path)
