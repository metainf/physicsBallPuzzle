"""This example spawns (bouncing) balls randomly on a L-shape constructed of
two segment shapes. Not interactive.
"""

__version__ = "$Id:$"
__docformat__ = "reStructuredText"

# Python imports
import random
import math
import argparse
import datetime

# Library imports
import pygame
from pygame.key import *
from pygame.locals import *
from pygame.color import *

# pymunk imports
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d

from shapely.geometry import Polygon
from shapely.geometry import Point

import numpy as np



class BouncyBalls(object):
    """
    This class implements a simple scene in which there is a static platform (made up of a couple of lines)
    that don't move. Balls appear occasionally and drop onto the platform. They bounce around.
    """
    def __init__(self,collision,replayFile,filename):
        # Space
        self._collision = collision
        self._space = pymunk.Space()
        self._space.gravity = (0.0, -900.0)

        # Physics
        # Time step
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        pygame.init()
        flags = DOUBLEBUF
        self._screenWidth = 800
        self._screenHeight = 800
        self._screen = pygame.display.set_mode((self._screenWidth, self._screenHeight),flags)
        self._screen.set_alpha(None)
        self._clock = pygame.time.Clock()
        self._ballRadius = 25.0
        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)
        self._draw_options.shape_outline_color = self._draw_options.shape_dynamic_color
        self._draw_options.flags ^= self._draw_options.DRAW_COLLISION_POINTS
        self._draw_options.flags ^= self._draw_options.DRAW_CONSTRAINTS
        self._windowSize = int(128)
        # Static polygon
        self._staticPolygon = None

        # Ball that exist in the world
        self._ball = None
        self._ballStart = (0,0)
        self._polyInfo = None

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()
        self._reset_scene(collision,replayFile)

        # Execution control and time until the next ball spawns
        self._running = True

        self._ticks = 0
        self._maxTicks = 200
        self.ballImgArray = np.zeros((self._windowSize,self._windowSize,3,self._maxTicks),dtype=np.uint8)
        self.velArray = np.zeros((2,self._maxTicks))
        self.filename = filename

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

            self._process_events()
            self._clear_screen()
            self._draw_objects()
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(60)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))
            screen = pygame.display.get_surface()
            screenNumpy = pygame.surfarray.pixels3d(screen)
            ballPos = list(self._ball.body.position.int_tuple)
            ballPos[1] = -ballPos[1] + self._screenHeight
            ballImg = screenNumpy[ballPos[0]-self._windowSize//2:ballPos[0]+self._windowSize//2,ballPos[1]-self._windowSize//2:ballPos[1]+self._windowSize//2,:]
            ballImg = np.transpose(ballImg,(1, 0, 2))
            ballVel = self._ball.body.velocity.int_tuple

            self.ballImgArray[:,:,:,self._ticks] =  ballImg
            self.velArray[:,self._ticks] = ballVel

            self._ticks += 1
            if self._ticks >= self._maxTicks:
                date = datetime.datetime.now()
                if self._collision:
                    self.filename += "C"
                else:
                    self.filename += "N"
                np.savez_compressed(self.filename,
                    ballImgArray=self.ballImgArray,
                    velArray=self.velArray,
                    ballStart = self._ballStart,
                    polygon = self._polyInfo)
                self._running = False


    def _add_static_scenery(self):
        """
        Create the static bodies.
        :return: None
        """
        static_body = self._space.static_body
        static_lines = [pymunk.Segment(static_body, (100, 100), (700, 100), 10),
                        pymunk.Segment(static_body, (700, 100), (700, 700), 10),
                        pymunk.Segment(static_body, (700, 700), (100, 700), 10),
                        pymunk.Segment(static_body, (100, 700), (100, 100), 10)]
        for line in static_lines:
            line.elasticity = 0.95
            line.friction = 0.50
        self._space.add(static_lines)

    def _create_ball(self,x,y):
        """
        Create a ball.
        :return:
        """
        if self._ball is None:
            mass = 10
            inertia = pymunk.moment_for_circle(mass, 0, self._ballRadius, (0, 0))
            body = pymunk.Body(mass, inertia)
            body.position = x, y
            self._ballStart = (x,y)
            shape = pymunk.Circle(body, self._ballRadius, (0, 0))
            shape.elasticity = 0.5
            shape.friction = 0.9
            self._space.add(body, shape)
            self._ball = shape


    def _reset_scene(self,collision,replayFile):
        # First reset the polygon and ball
        if self._staticPolygon is not None:
            self._space.remove(self._staticPolygon)
            self._staticPolygon = None
        if self._ball is not None:
            self._space.remove(self._ball)
            self._ball = None

        if replayFile is None:
            # Pick a location for the ball the start in the upper half of the space
            ballx = random.randint(100+self._ballRadius,700-self._ballRadius)
            bally = random.randint(100+300,700-self._ballRadius)
            self._create_ball(ballx,bally)
            ballPoly = Point(ballx, bally).buffer(self._ballRadius)
            ballRect = Polygon([(ballx-self._ballRadius,bally-self._ballRadius),
                                (ballx+self._ballRadius,bally-self._ballRadius),
                                (ballx-self._ballRadius,100),
                                (ballx+self._ballRadius,100)])
            foundValidPoly = False
            while(not foundValidPoly):
                b = pymunk.Body(body_type=pymunk.Body.STATIC)
                b.position = random.randint(100+100,700-100),random.randint(100+100,bally-self._ballRadius)
                b.angle = random.uniform(0, math.pi * 2.0)
                static_poly = pymunk.Poly(b,[(0,0),(0,100),(100,0)])
                polyPointsWorld = []
                for v in static_poly.get_vertices():
                    x,y = v.rotated(static_poly.body.angle) + static_poly.body.position
                    polyPointsWorld.append((x,y))
                staticPolyShape = Polygon(polyPointsWorld)
                if(collision and staticPolyShape.intersects(ballRect) and not staticPolyShape.intersects(ballPoly)):
                    foundValidPoly = True
                elif(not collision and not staticPolyShape.intersects(ballRect) and not staticPolyShape.intersects(ballPoly)):
                    foundValidPoly = True
                if(foundValidPoly):
                    static_poly.elasticity = .95
                    static_poly.friction = .50
                    self._staticPolygon = static_poly
                    self._space.add(static_poly)
                    self._polyInfo = (b.position,b.angle,static_poly.get_vertices())
        else:
            loaded = np.load(replayFile)
            ballPos = loaded['ballStart']
            self._create_ball(ballPos[0],ballPos[1])
            self._polyInfo = loaded['polygon']
            b = pymunk.Body(body_type=pymunk.Body.STATIC)
            b.position = self._polyInfo[0]
            b.angle = self._polyInfo[1]
            static_poly = pymunk.Poly(b,self._polyInfo[2])
            static_poly.elasticity = .95
            static_poly.friction = .50
            self._staticPolygon = static_poly
            self._space.add(static_poly)


    def _process_events(self):
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == QUIT:
                self._running = False


    def _clear_screen(self):
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(THECOLORS["white"])


    def _draw_objects(self):
        """
        Draw the objects.
        :return: None
        """
        self._space.debug_draw(self._draw_options)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collision", help="set collision",action="store_true")
    parser.add_argument("-r", "--replay",help="replay's a file")
    parser.add_argument("name")
    args = parser.parse_args()
    game = BouncyBalls(args.collision,args.replay,args.name)
    game.run()
