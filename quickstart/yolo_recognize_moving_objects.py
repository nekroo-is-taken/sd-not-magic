"""
1. Create a main_vehicle and camera
2. Create random cars in the map
3. Use YOLO try to recognize them.
"""

import random
import subprocess
from pathlib import Path

import cv2
import torch
import numpy as np
from absl import app

import carla

SYNC_MASTER = True
USE_CUDA_IN_FFMPEG = True  # Set 'False' if you don't need CUDA.
MAX_NUM = 1000  # Number to take pictures from camera.
IMAGE_FOLDER = '../tmp'
OUT_VIDEO_RAW = f'{IMAGE_FOLDER}/{Path(__file__).stem}.mp4'
# Set output path for the video
# Processed video will be stored under 'YOLO_VIDEO_PROJECT/YOLO_VIDEO_FOLDER'
YOLO_VIDEO_PROJECT = '..'
YOLO_VIDEO_FOLDER = 'out'
YOLO_DETECT = Path('../../../common/yolov5/detect.py')
random.seed(2)

model = None  # YOLO model
count = 0  # Taken picture count
actor_list = []
world = None  # Carla world object

def show_image(carla_img):
    """Stream the view from camera."""
    global count
    carla_img.save_to_disk(f'{IMAGE_FOLDER}/{count:04}.png')
    count += 1
    if count % 100 == 0:
        print(f'{count} pieces of image have been saved.')


def make_video(input_dir, raw_video, yolo_project, yolo_folder):
    subprocess.check_output(
        f'ffmpeg {"-hwaccel cuda" if USE_CUDA_IN_FFMPEG else ""} -y -r 25 -s 800x600 -i {input_dir}/%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p {raw_video}',
        shell=True)
    subprocess.check_output(
        f'python {YOLO_DETECT} --source {raw_video} --weights yolov5l6.pt --conf 0.25 --project {yolo_project} --name {yolo_folder} --exist-ok',
        shell=True)

def spawn_npc():
    global actor_list
    global world

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # Create vehicles in random spawn points
    NUM_OF_VEHS = 100
    for i in range(0, NUM_OF_VEHS):
        # randomly pick a car and its position in the map
        bp = random.choice(blueprint_library.filter('vehicle'))
        init_main_transform = random.choice(spawn_points)

        # This time we are using `try_spawn_actor()`. If the spot is already
        # occupied by another object, the function will return None.
        npc = world.try_spawn_actor(bp, init_main_transform)
        if npc is not None:
            actor_list.append(npc)
            npc.set_autopilot(True)
            print('created %s' % npc.type_id)


def init_yolo():
    global model
    # MODEL_STORAGE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'model', 'pretrained')
    # torch.hub.set_dir(MODEL_STORAGE_PATH)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True)


def main(argv):
    global actor_list
    global world

    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        host_ip = '127.0.0.1'
        client = carla.Client(host_ip, 2000)
        client.set_timeout(5.0)

        # Once we have a client we can retrieve the world that is currently running.
        world = client.get_world()
        tm = client.get_trafficmanager()
        settings = world.get_settings()

        if SYNC_MASTER:
            settings.fixed_delta_seconds = 0.04
            settings.synchronous_mode = SYNC_MASTER  # (Required) Enables synchronous mode on world
            tm.set_synchronous_mode(
                SYNC_MASTER
            )  # (Required) Enables synchronous mode on traffic manager
            world.apply_settings(settings)

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()

        #NOTE: the main vehicle is deterministic now
        bp = blueprint_library.find('vehicle.tesla.model3')
        main_vehicle_transform = random.choice(
            world.get_map().get_spawn_points())
        main_vehicle = world.spawn_actor(bp, main_vehicle_transform)
        main_vehicle.set_autopilot(True)
        actor_list.append(main_vehicle)
        print('created %s' % main_vehicle.type_id)

        # Let's add now a "depth" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        # The vehicle model towards positive "x". Positive "z" is upward.
        camera_transform = carla.Transform(carla.Location(x=2.3, z=1.2))
        camera = world.spawn_actor(camera_bp,
                                   camera_transform,
                                   attach_to=main_vehicle)
        actor_list.append(camera)
        # When sensor gets data, it will process with `show_image()`
        camera.listen(lambda data: show_image(data))
        print('created %s' % camera.type_id)

        # Prepare npcs to recognize
        spawn_npc()

        # Last, init YOLO model
        init_yolo()

        while True:
            if count > MAX_NUM:
                break
            world.tick()

    finally:
        print('Disconnecting from server...')
        if SYNC_MASTER:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        print('Destroying actors')
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('Done.')

    # Collect processed images and make it a video
    make_video(input_dir=IMAGE_FOLDER, raw_video=OUT_VIDEO_RAW, yolo_project=YOLO_VIDEO_PROJECT, yolo_folder=YOLO_VIDEO_FOLDER)


if __name__ == '__main__':
    app.run(main)
