import os
import shutil

import cv2

# convert RGB to Grayscale
seq = 'ME000'
input_folder = '/datasets/TartanAirMono/' + seq
output_folder = '/datasets/TartanAirMono_grayscale/' + seq + '/img'
if os.path.isdir(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# create a timestamp in nanoseconds
dt = 100000000
ts = 1e12
ts_list = []
n = 0

for filename in sorted(os.listdir(input_folder)):
    input_path = os.path.join(input_folder, filename)
    
    out_filename = str(int(ts)) + '.png'
    output_path = os.path.join(output_folder, out_filename)

    image = cv2.imread(input_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(output_path, gray_image)
        
    ts_list.append(ts)
    ts += dt
    
    n += 1

print('saved %d images to %s' % (n, output_folder))