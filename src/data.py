import os
import cv2
import numpy as np

from xml.etree import ElementTree




class_names = ['person']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

n_classes = 1
size = (640,640)


def load_data(data_dir: str=None):

    datasets = ['train', 'test', 'valid']
    output = []

    for dataset in datasets:
        imags = []
        labels = []
        directoryA = data_dir + dataset +"/labels"
        directoryIMG = data_dir + dataset +"/images/"
        file = os.listdir(directoryA)
        img = os.listdir(directoryIMG)
        file.sort()
        img.sort()
        print(file)
        print(img)
        i=0
        for xml in file:

            xmlf = os.path.join(directoryA,xml)
            print(directoryA)
            print(xml)
            dom = ElementTree.parse(xmlf)
            vb = dom.findall('object')
            label = vb[0].find('name').text
            labels.append(class_names_label[label])

            img_path = directoryIMG + img[i]
            curr_img = cv2.imread(img_path)
            curr_img = cv2.resize(curr_img, size)
            imags.append(curr_img)
            i +=1
        
        imags = np.array(imags, dtype='float32')
        imags = imags / 255
        
      #  labels = pd.DataFrame(labels)
        labels = np.array(labels, dtype='int32')

        output.append((imags, labels))
        print('yesssss\n\nyess!!!')
    return output
