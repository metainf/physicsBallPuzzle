import copy
from PIL import Image, ImageDraw

import pymunk
from pymunk import Vec2d
import ImgToObj

class PymunkSim(object):
  def __init__(self):
    # Space
    self._space = pymunk.Space()
    self._space.gravity = (0.0, -35.0)
    self._elasticity = .3
    self._friction = .6

    # Physics
    # Time step
    self._dt = 1.0 / 60.0
    self._max_sim_time = 17.0
    self._ticks = 0
    self._max_ticks = int(self._max_sim_time/self._dt)
    self._contact_time = 3.0


    # Static barrier walls (lines) that the balls bounce off of
    self._add_walls()

  def run(self):
    """
    The main loop of the game.
    :return: None
    """
    # Main loop
    while self._running:
        # Progress time forward
        for x in range(self._physics_steps_per_frame):
            self._space.step(self._dt)

  def _add_walls(self):
    """
    Create the static walls.
    :return: None
    """
    static_body = space.static_body
    static_lines = [pymunk.Segment(static_body, (0.0, 0.0), (0.0, 256.0), 0.0),
                    pymunk.Segment(static_body, (0.0, 256.0), (256.0, 256.0), 0.0),
                    pymunk.Segment(static_body, (256.0, 256.0), (256.0, 0.0), 0.0),
                    pymunk.Segment(static_body, (256.0, 0.0), (0.0, 0.0), 0.0)]
    for line in static_lines:
      line.elasticity = self._elasticity
      line.friction = self._friction
    space.add(static_lines)

    def init_scene(self,scene_objects):
      self._scene_objects = copy.deepcopy(scene_objects)
      # Add the scene objects to the space
      for i,layer_objects in enumerate(self._scene_objects):
        # Add the balls
        layer_objects['balls'] = []
        for circle in layer_objects['circles']:
          radius = circle[2]
          body = pymunk.Body()
          if i == ImgToObj.Layer.static_goal.value or i == ImgToObj.Layer.static_body.value:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
          body.position = circle[0],circle[1]
          shape = pymunk.Circle(body, radius, Vec2d(0,0))
          shape.elasticity = 0.3
          shape.friction = 0.6
          shape.density = 1
          if i == ImgToObj.Layer.object.value:
            shape.collision_type = COLLISION_OBJECT
          elif i == ImgToObj.Layer.dynamic_goal.value or i == ImgToObj.Layer.static_goal.value:
            shape.collision_type = COLLISION_GOAL
          space.add(body, shape)
          layer_objects['balls'].append((shape,[body.position]))

        # Add the polygons to the space
        # A list of lists of triangles per polygon
        layer_objects['pymunk_polys'] = []
        for polygon in layer_objects['polygons']:
          triangles = []
          center = np.mean(polygon[1]['vertices'].astype(float),axis=0)
          body = pymunk.Body()
          if i == ImgToObj.Layer.static_goal.value or i == ImgToObj.Layer.static_body.value:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
          body.position = center
          for triangle in polygon[1]['triangles'].tolist():
            shape = pymunk.Poly(body,polygon[1]['vertices'][triangle,:].astype(float)-center,radius=0.25)
            shape.elasticity = 0.3
            shape.friction = 0.6
            shape.density = 1
            if i == ImgToObj.Layer.object.value:
              shape.collision_type = COLLISION_OBJECT
            elif i == ImgToObj.Layer.dynamic_goal.value or i == ImgToObj.Layer.static_goal.value:
              shape.collision_type = COLLISION_GOAL
            space.add(shape)
            triangles.append(shape)
          space.add(body)
          layer_objects['pymunk_polys'].append(triangles)

    def _draw_objects(self):
      """
      Draw the scene.
      """
      img = Image.new('RGB', (256,256), (255, 255, 255))
      draw = ImageDraw.Draw(img)
      for i,layer_objects in enumerate(self._scene_objects):

        for ballData in layer_objects['balls']:
          circle1=plt.Circle((ballData[0].body.position.x,256.0-ballData[0].body.position.y),radius=ballData[0].radius,color='b',fill=True)
          axs[1].add_artist(circle1)

        for pymunk_poly in layer_objects['pymunk_polys']:
          for triangle in pymunk_poly:
            verts = []
            for v in triangle.get_vertices():
              x,y = v.rotated(triangle.body.angle) + triangle.body.position
              verts.append([x,y])
            verts = np.array(verts)
            verts[:,1] = 256.0-verts[:,1]
            polygon = Polygon(verts,color='b')
            patches.append(polygon)

