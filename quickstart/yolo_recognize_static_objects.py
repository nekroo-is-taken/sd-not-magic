"""
1. Create a main_vehicle
2. Create a camera and attach to main_vehicle
3. Streaming in cv2.imshow().
"""

import random
import os

import cv2
import torch
import numpy as np
from absl import app

import carla

SYNC_MASTER = True
MAX_NUM = 2
random.seed(10)

# YOLO model
model = None
count = 0

def show_image(carla_img):
  """Stream the view from camera."""
  global count 
  global model
  count += 1
  np_img = np.array(carla_img.raw_data, dtype=np.uint8).reshape((600, 800, 4))#[:,:,:3]
  np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
  res = model(np_img)
  res.show()

def init_yolo():
  global model
  # MODEL_STORAGE_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'model', 'pretrained')
  # torch.hub.set_dir(MODEL_STORAGE_PATH)
  model = torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True)

def main(argv):
  actor_list = []

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
      tm.set_synchronous_mode(SYNC_MASTER)  # (Required) Enables synchronous mode on traffic manager
      world.apply_settings(settings)

    # The world contains the list blueprints that we can use for adding new
    # actors into the simulation.
    blueprint_library = world.get_blueprint_library()

    #NOTE: the main vehicle is deterministic now
    bp = blueprint_library.find('vehicle.tesla.model3')
    main_vehicle_transform = random.choice(world.get_map().get_spawn_points())
    main_vehicle = world.spawn_actor(bp, main_vehicle_transform)
    actor_list.append(main_vehicle)
    print('created %s' % main_vehicle.type_id)

    # Let's add now a "depth" camera attached to the vehicle. Note that the
    # transform we give here is now relative to the vehicle.
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    # The vehicle model towards positive "x". Positive "z" is upward. 
    camera_transform = carla.Transform(carla.Location(x=1.2, z=1.2))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=main_vehicle)
    actor_list.append(camera)
    # When sensor gets data, it will process with `show_image()`
    camera.listen(lambda data: show_image(data))
    print('created %s' % camera.type_id)
    
    # Create vehicles in front of the `main_vehicle`
    front_car_bp = blueprint_library.find('vehicle.nissan.patrol')
    front_car_transform = main_vehicle_transform
    front_car_transform.location.y += -6
    front_car_transform.location.z += 0.1
    front_car = world.spawn_actor(front_car_bp, front_car_transform)
    actor_list.append(front_car)
    print('created %s' % front_car.type_id)
      
    # Last, init YOLO model
    init_yolo()
    
    while True:
      if count >= MAX_NUM:
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
    cv2.destroyAllWindows()
    camera.destroy()
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    print('Done.')

if __name__ == '__main__':
  app.run(main)
