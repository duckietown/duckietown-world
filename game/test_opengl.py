from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.raw.GLU import gluBuild2DMipmaps

from game1 import MapDistribution, create_map, sample_cars, get_background_image

w, h = 500, 500


def square():
    glBegin(GL_QUADS)
    glVertex2f(100, 100)
    glVertex2f(200, 100)
    glVertex2f(200, 200)
    glVertex2f(100, 200)
    glEnd()


a = 0


def iterate():
    global a
    a += 0.1
    glViewport(0, 0, 500, 500)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0.0, 500, 0.0, 500 + a, 0.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def showScreen():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    iterate()
    glColor3f(1.0, 0.0, 3.0)
    square()
    glutSwapBuffers()


import numpy as np


def get_texture(img: np.ndarray):
    h, w = img.shape[:2]

    glEnable(GL_TEXTURE_2D)
    textureID = glGenTextures(1)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
    glBindTexture(GL_TEXTURE_2D, textureID)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, h, w, 0, GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
    return textureID


def go():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA)
    glutInitWindowSize(500, 500)
    glutInitWindowPosition(0, 0)

    window_name = "wn"
    md = MapDistribution(n=500, min_d=50, max_d=80, width_road=10)
    wm = create_map(md)
    N = 200
    N_min = 100
    cars = sample_cars(wm, n=N)
    bg = get_background_image(wm)

    textureID = get_texture(bg)

    # texture = glGenTextures(1)
    # glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    # glBindTexture(GL_TEXTURE_2D, texture)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    # gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, h, w, GL_RGB, GL_UNSIGNED_BYTE, bg)

    #
    # glTexImage2D(GL_TEXTURE_2D,
    #              0,
    #              GL_RGB,
    #              w, h,
    #              0,
    #              GL_RGB,
    #              GL_UNSIGNED_BYTE,
    #              bg)

    it = 0

    frame = bg.copy()

    wind = glutCreateWindow("OpenGL Coding Practice")
    glutDisplayFunc(showScreen)
    glutIdleFunc(showScreen)

    # while True:
    #     it += 1
    #     update_cars(wm, cars, dt=Decimal(1), min_cars=N_min)
    #
    #     if it % 1 == 0:
    #
    #         ncars = 10
    #         pretties = []
    #         for k in list(cars)[:ncars]:
    #             if not cars[k].obs_history:
    #                 continue
    #             pretty = pretty_obs(cars[k].obs_history[-1])
    #             pretties.append(np.ones((pretty.shape[0], 5, 3), 'uint8') * 255)
    #             pretties.append(pretty)
    #
    #         rgb = np.hstack(tuple(pretties))
    #         cv2.imshow("one", rgb)
    #
    #         draw_cars(bg, frame, cars, erase=False)
    #
    #         cv2.imshow(window_name, frame)
    #         draw_cars(bg, frame, cars, erase=True)
    #         cv2.waitKey(1)

    # print(it)
    # cv2.destroyAllWindows()
    glutMainLoop()


if __name__ == "__main__":
    go()
