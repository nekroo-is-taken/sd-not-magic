"""
1. Create a main_vehicle
2. Create a camera and attach to main_vehicle
3. Streaming in cv2.imshow().
"""

import random

import cv2
import numpy as np
from absl import app

import carla

def show_image(carla_img):
  np_img = np.array(carla_img.raw_data, dtype=np.uint8).reshape((600, 800, 4))[:,:,:3]
  cv2.imshow("Stream", np_img)
  cv2.waitKey(1)

def main(argv):
  actor_list = []

  try:
    # First of all, we need to create the client that will send the requests
    # to the simulator. Here we'll assume the simulator is accepting
    # requests in the localhost at port 2000.
#     host_ip = os.environ['host_ip']
    host_ip = '127.0.0.1'
    client = carla.Client(host_ip, 2000)
    client.set_timeout(5.0)

    # Once we have a client we can retrieve the world that is currently running.
    world = client.get_world()
    tm = client.get_trafficmanager()

    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.04
    settings.synchronous_mode = True  # (Required) Enables synchronous mode on world
    tm.set_synchronous_mode(True)  # (Required) Enables synchronous mode on traffic manager
    world.apply_settings(settings)

    # The world contains the list blueprints that we can use for adding new
    # actors into the simulation.
    blueprint_library = world.get_blueprint_library()

    # Now let's filter all the blueprints of type 'vehicle' and choose one
    # at random.
    bp = random.choice(blueprint_library.filter('vehicle'))

    # A blueprint contains the list of attributes that define a vehicle's
    # instance, we can read them and modify some of them. For instance,
    # let's randomize its color.
    if bp.has_attribute('color'):
      color = random.choice(bp.get_attribute('color').recommended_values)
      bp.set_attribute('color', color)

    # Now we need to give an initial transform to the vehicle. We choose a
    # random transform from the list of recommended spawn points of the map.
    init_main_transform = random.choice(world.get_map().get_spawn_points())
    # So let's tell the world to spawn the vehicle.
    main_vehicle = world.spawn_actor(bp, init_main_transform)
    # Let's put the vehicle to drive around.
    main_vehicle.set_autopilot(True)
    
    # It is important to note that the actors we create won't be destroyed
    # unless we call their "destroy" function. If we fail to call "destroy"
    # they will stay in the simulation even after we quit the Python script.
    # For that reason, we are storing all the actors we create so we can
    # destroy them afterwards.
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
    print('created %s' % camera.type_id)

    # When sensor gets data, it will process with `show_image()`
    camera.listen(lambda data: show_image(data))
    
    import time
    while True:
      time.sleep(0.04)
      world.tick()
    
  finally:
    print('destroying actors')
    camera.listen(lambda data: data)
    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    cv2.destroyWindow("Stream")
    print('done.')

if __name__ == '__main__':
  app.run(main)
