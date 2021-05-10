import numpy as np
import json
import math
import os
from tqdm import tqdm


data_list = json.load(open('pix3d.json'))

csv_file = 'Pix3D.txt'
if not os.path.exists(csv_file):
    with open(csv_file, 'a') as f:
        f.write('source,set,cls_name,im_name,object,cad_index,truncated,occluded,difficult,azimuth,elevation,inplane_rotation,left,upper,right,lower,has_keypoints,distance,im_path\n')

for i in tqdm(range(0, len(data_list))):
    img_source = data_list[i]['img_source']
    model_source = data_list[i]['model_source']
    truncated = data_list[i]['truncated']
    occluded = data_list[i]['occluded']
    slightly_occluded = data_list[i]['slightly_occluded']
    occluded = occluded or slightly_occluded

    img_path = data_list[i]['img']
    cls_name = data_list[i]['category']

    # retrieve 3D model name
    cad_index = data_list[i]['model']

    # retrieve bbox locations
    [left, upper, right, lower] = data_list[i]['bbox']

    # compute elevation and azimuth from cam_position
    cam_pose = data_list[i]['cam_position']
    elevation = math.degrees(math.atan(cam_pose[1] / math.sqrt(cam_pose[0]**2 + cam_pose[2]**2)))
    cam_pose[0] = -cam_pose[0]
    if cam_pose[2] == 0. or cam_pose[2] == -0.:
        if cam_pose[0] <= 0:
            azimuth = 90.
        else:
            azimuth = 270.
    else:
        azimuth = math.degrees(math.atan(cam_pose[0] / cam_pose[2]))
        if cam_pose[0] <= 0 and cam_pose[2] < 0:
            azimuth = azimuth
        elif cam_pose[0] > 0 and cam_pose[2] < 0:
            azimuth = 360. + azimuth
        else:
            azimuth = 180. + azimuth

    azimuth = (360 - azimuth) % 360
    inplane_rotation = data_list[i]['inplane_rotation']
    inplane_rotation = -inplane_rotation * 180 / np.pi

    with open(csv_file, 'a') as f:
        f.write(img_source + ',' + 'val' + ',' + cls_name + ',' + img_path + ',' + str(0) + ',' + cad_index + ',' +
                str(truncated) + ',' + str(occluded) + ',' + str(0) + ',' +
                str(azimuth) + ',' + str(elevation) + ',' + str(inplane_rotation) + ',' +
                str(left) + ',' + str(upper) + ',' + str(right) + ',' + str(lower) + ',' +
                str(1) + ',' + str(0) + ',' + img_path + '\n')
