import glob
import os
import random
import cv2
import threading

from shutil import copyfile
from subprocess
from xml.etree import ElementTree as ET

# Paths and configurations
current_dir = os.path.joih(os.getcwd(),'Webcam_Face_Detection') 
xmlFolder = os.path.join(current_dir,'yolo','lables_test')
imgFolder = os.path.join(current_dir,'yolo','images_test')
saveYoloPath = os.path.join(current_dir,'yolo','model')
classList = {"face":0}

modelYOLO = 'yolov3-tiny'  #yolov3 or yolov3-tiny
testRatio = 0.2
cfgFolder = 'cfg.face'
cfg_obj_names = 'obj.names'
cfg_obj_data = 'obj.data'

negative_images = True  # images with no xml files as negative images
numBatch = 24
numSubdivision = 3
darknetEcec = 'D:\\darknet\\darknet-master\\'

# check dir folder exists
def check_dir(directory) :
  if not os.path.exists(directory):
      os.makedirs(directory)
    
check_dir(saveYoloPath)
  
# Transfer VOC data to YOLO format
def transferYolo(xmlFilepath, imgFilepath, labelGrep=''):

    img_file, img_file_extension = os.path.splitext(imgFilepath)
    img_filename = os.path.basename(img_file)
    yoloFilename = os.path.join(saveYoloPath ,img_filename + '.txt')

    if xmlFilepath or negative_images is False:
        try :
          img = cv2.imread(imgFilepath)
          if img is None:
            raise FileNotFoundError(f'Image file {imgFilepath} not found.')
          imgShape = img.shape
        
          img_h = imgShape[0]
          img_w = imgShape[1]

        tree = ET.parse(xmlFilepath)
        root = tree.getroot()

         with open(yoloFilename, 'w') as the_file:
                for obj in root.findall('object'):
                    className = obj.find('name').text
                    if className == labelGrep or not labelGrep:
                        classID = classList.get(className, -1)
                        if classID == -1:
                            continue

                        bbox = obj.find('bndbox')
                        xmin = int(bbox.find('xmin').text)
                        ymin = int(bbox.find('ymin').text)
                        xmax = int(bbox.find('xmax').text)
                        ymax = int(bbox.find('ymax').text)

                        x = (xmin + (xmax - xmin) / 2) / img_w
                        y = (ymin + (ymax - ymin) / 2) / img_h
                        w = (xmax - xmin) / img_w
                        h = (ymax - ymin) / img_h

                        the_file.write(f"{classID} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        except Exception as e:
            print(f'Error processing file {xmlFilepath}: {e}')

# Transfer VOC dataset to YOLO dataset

fileCount = 0
if negative_images :
    print('Images without xmlfiles will be treated as negative images.')

for file in os.listdir(imgFolder):
    
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()
    
    if file_extension in ['.jpg', '.png', '.jpeg', '.bmp'):
        imgfile = os.path.join(imgFolder ,file)
        xmlfile = os.path.join(xmlFolder ,filename + '.xml')

        if os.path.isfile(xmlfile) :
            fileCount += 1
            transferYolo(xmlfile, imgfile, '')
            copyfile(imgfile, os.path.join(saveYoloPath ,file))

        elif negative_images is True :
            transferYolo(None, imgfile, '')
            copyfile(imgfile, os.path.join(saveYoloPath ,file))

print(f'{fileCount} images transferred.')


# Create YOLO cfg folder and split dataset to train and test datasets
check_dir(cfgFolder)

fileList = [os.path.join(saveYoloPath ,file) for file in os.listdir(saveYoloPath) if os.path.splitext(file)[1].lower() in ['.jpg', '.png', '.jpeg', '.bmp']]
random.shuffle(fileList)
trainCount = int(len(fileList) * (1 - testRatio))
train_data = fileList[:fileList]
test_data = fileList[fileList:]

with open(os.path.join(cfgFolder,'train.txt'), 'w') as the_file:
    for i in train_data:
        the_file.write(fileList[i] + '\n')
the_file.close()

with open(os.path.join(cfgFolder,'test.txt'), 'w') as the_file:
    for i in test_data:
        the_file.write(fileList[i] + '\n')
the_file.close()

print(f'Train dataset:{len(train_data)} images')
print(f'Test dataset:{len(test_data)} images')

# Generate data & names files  
check_dir(os.path.join(cfgFolder,'weights'))

with open(os.path.join(cfgFolder ,cfg_obj_data), 'w') as the_file:
    the_file.write('classes= ' + str(len(classList)) + '\n')
    the_file.write('train  = ' + os.path.join(cfgFolder ,'train.txt') + '\n')
    the_file.write('valid  = ' + os.path.join(cfgFolder ,'test.txt') + '\n')
    the_file.write('names = ' + os.path.join(cfgFolder ,'obj.names') + '\n')
    the_file.write('backup = ' + os.path.join(cfgFolder ,'weights'))

with open(os.path.join(cfgFolder ,cfg_obj_names), 'w') as the_file:
    for className in classList:
        the_file.write(className + '\n')


# update YOLO config file
classNum = len(classList)
filterNum = (classNum + 5) * 3
fileCFG = 'yolov3.cfg' if modelYOLO == 'yolov3' else 'yolov3-tiny.cfg'

with open(os.path.join(darknetEcec,'cfg',fileCFG),'r') as file:
  file_content = file.read()
  
file_content = file_content.replace('{BATCH}', str(numBatch))
file_content = file_content.replace('{SUBDIVISIONS}', str(numSubdivision))
file_content = file_content.replace('{FILTERS}', str(filterNum))
file_content = file_content.replace('{CLASSES}', str(classNum))

with open(os.path.join(cfgFolder,fileCFG), 'w') as file:
  file.write(file_updated)

# Start to train the YOLO model

executeCmd = darknetEcec + "darknet.exe detector train " +os.path.join(os.getcwd(),cfgFolder,"obj.data") + " " \
    + os.path.join(os.getcwd(),cfgFolder,fileCFG) + " darknet53.conv.74"

timeout_seconds = 20*60*60 # limit max training time

def run_command():
  process = subprocess.Popen(executeCmd.split(), shell=True)
  return process

try:
  process = run_command()
  thread = threading.Thread(process.wait)
  thread.start()

  thread.join(timeout_seconds)
  
  if thread.is_alive():
    print("Command timed out. Terminating...")
    os.system('taskkill /f /im darknet.exe')
# press ctrl+c to stop the thread
except KeyboardInterrupt:
  print("Process interrupted by user (Ctrl+C).")
  os.system('taskkill /f /im darknet.exe')
  
print('after training, you can find all the weights files here:' + os.path.join(cfgFolder ,'weights'))
