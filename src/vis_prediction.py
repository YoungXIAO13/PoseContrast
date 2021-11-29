import os
import bpy
import argparse
import math
from math import radians
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
import numpy as np
import pandas as pd
import cv2
import pickle


# Crop and Resize the image and paste it to the bg
def crop_resize_paste(im, bg, left, upper, right, lower):
    # crop the RGBA image according to alpha channel
    bbox = im.getbbox()
    im = im.crop(bbox)

    # resize & padding the rendering then paste on the bg
    w, h = im.size
    target_w, target_h = right - left, lower - upper
    ratio = min(float(target_w) / w, float(target_h / h))
    new_size = (int(w * ratio), int(h * ratio))
    im = im.resize(new_size, Image.BILINEAR)
    bg.paste(im, (left + (target_w - new_size[0]) // 2, upper + (target_h - new_size[1]) // 2))


# create a lamp with an appropriate energy
def makeLamp(lamp_name, rad):
    # Create new lamp data block
    lamp_data = bpy.data.lamps.new(name=lamp_name, type='POINT')
    lamp_data.energy = rad
    # modify the distance when the object is not normalized
    # lamp_data.distance = rad * 2.5

    # Create new object with our lamp data block
    lamp_object = bpy.data.objects.new(name=lamp_name, object_data=lamp_data)

    # Link lamp object to the scene so it'll appear in this scene
    scene = bpy.context.scene
    scene.objects.link(lamp_object)
    return lamp_object


def parent_obj_to_camera(b_camera):
    # set the parenting to the origin
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


def clean_obj_lamp_and_mesh(context):
    scene = context.scene
    objs = bpy.data.objects
    meshes = bpy.data.meshes
    for obj in objs:
        if obj.type == "MESH" or obj.type == 'LAMP':
            scene.objects.unlink(obj)
            objs.remove(obj)
    for mesh in meshes:
        meshes.remove(mesh)


def render_obj(obj, output_dir, azi, ele, rol, name, shape=[512, 512], forward=None, up=None):
    clean_obj_lamp_and_mesh(bpy.context)

    # Setting up the environment
    scene = bpy.context.scene
    scene.render.resolution_x = shape[1]
    scene.render.resolution_y = shape[0]
    scene.render.resolution_percentage = 100
    scene.render.alpha_mode = 'TRANSPARENT'

    # Camera setting
    cam = scene.objects['Camera']
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty

    # Light setting
    lamp_object = makeLamp('Lamp1', 5)
    lamp_add = makeLamp('Lamp2', 1)

    if forward is not None and up is not None:
        bpy.ops.import_scene.obj(filepath=obj, axis_forward=forward, axis_up=up)
    else:
        bpy.ops.import_scene.obj(filepath=obj)

    # normalize it and set the center
    for object in bpy.context.scene.objects:
        if object.name in ['Camera', 'Lamp'] or object.type == 'EMPTY':
            continue
        bpy.context.scene.objects.active = object
        max_dim = max(object.dimensions)
        object.dimensions = object.dimensions / max_dim if max_dim != 0 else object.dimensions

    # Output setting
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = os.path.join(output_dir, name)
    # scene.render.filepath = os.path.join(output_dir, name + '_render_%03d_%03d_%03d' % (int(azi), int(ele), int(rol)))

    # transform Euler angles from degrees into radians
    azi = radians(azi)
    ele = radians(ele)
    rol = radians(rol)
    r = 3
    loc_y = r * math.cos(ele) * math.cos(azi)
    loc_x = r * math.cos(ele) * math.sin(azi)
    loc_z = r * math.sin(ele)
    cam.location = (loc_x, loc_y, loc_z + 0.5)
    lamp_object.location = (loc_x, loc_y, 10)
    lamp_add.location = (loc_x, loc_y, -10)

    # Change the in-plane rotation
    cam_ob = bpy.context.scene.camera
    bpy.context.scene.objects.active = cam_ob  # select the camera object
    distance = np.sqrt(loc_x ** 2 + loc_y ** 2 + loc_z ** 2)
    bpy.ops.transform.rotate(value=rol, axis=(loc_x / distance, loc_y / distance, loc_z / distance),
                             constraint_axis=(False, False, False), constraint_orientation='GLOBAL', mirror=False,
                             proportional='DISABLED', proportional_edit_falloff='SMOOTH',
                             proportional_size=1)

    bpy.ops.render.render(write_still=True)


# =================Main Function======================= #

parser = argparse.ArgumentParser()
parser.add_argument('result', type=str, help='testing result path')
parser.add_argument('--root', type=str, default='data/Pix3D')
parser.add_argument('--gt', type=str, default='data/Pix3D/Pix3D.txt', help='gt csv file')
parser.add_argument('--cls', type=str, default='chair', help='class to evaluate')
parser.add_argument('--out', type=str, default='VisualResults/Pix3D')
args = parser.parse_args()


with open(os.path.join(args.result, 'correlation', '{}.pkl'.format(args.cls)), 'rb') as f:
    d = pickle.load(f)
test_errs = d['err']

# load gt and pred
frame = pd.read_csv(args.gt)
frame = frame[frame.set == 'val']
frame = frame[frame.truncated == 0]
frame = frame[frame.occluded == 0]
frame = frame[frame.has_keypoints == 1]
frame = frame[frame.elevation != 90]
frame = frame[frame.difficult == 0]

df_cls = frame[frame.cls_name == args.cls]
pred_cls = np.load(os.path.join(args.result, 'prediction', '{}.npy'.format(args.cls)))
assert pred_cls.shape[0] == len(df_cls)

out_dir = os.path.join(args.out, args.cls)
os.makedirs(out_dir, exist_ok=True)

# iterate on images
for idx in tqdm(range(len(df_cls))):
    img_pth = df_cls.iloc[idx]['im_path']
    img_name = img_pth.split('/')[-1].split('.')[0]
    img = cv2.imread(os.path.join(args.root, img_pth))

    # draw bounding box
    left = df_cls.iloc[idx]['left']
    upper = df_cls.iloc[idx]['upper']
    right = df_cls.iloc[idx]['right']
    lower = df_cls.iloc[idx]['lower']
    bbox = (left, upper, right, lower)

    # render viewpoint predictions
    [azi, ele, inp] = pred_cls[idx, :]
    save_name = '{}_PoseError_{:.2f}.png'.format(img_name, test_errs[idx])
    obj_pth = os.path.join(args.root, df_cls.iloc[idx]['cad_index'])
    render_obj(obj_pth,
               out_dir,
               azi, ele - 90, inp - 180,
               save_name,
               [521, 512], None, None)

    # blend viewpoint results into the original image
    w, h = img.shape[1], img.shape[0]
    bg = Image.new("RGBA", (w, h))
    vp_rend = Image.open(os.path.join(out_dir, save_name))
    crop_resize_paste(vp_rend, bg, left, upper, right, lower)

    draw = ImageDraw.Draw(bg)
    draw.rectangle([(left, upper), (right, lower)], outline=(0, 255, 0, 255), width=int(0.02 * min(img.shape[0], img.shape[1])))

    bg.save(os.path.join(out_dir, save_name))

