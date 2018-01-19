import numpy as np
from vispy import app, gloo
import Geometry
import random
from vispy.util.transforms import perspective, translate, rotate, ortho
from OpenGL.GLU import *
import math
from glm import project, unProject, GLM_DEPTH_CLIP_SPACE, GLM_DEPTH_ZERO_TO_ONE, tvec3

# Force using qt and take QtCore+QtGui from backend module
try:
    app_object = app.use_app('pyqt4')
except Exception:
    app_object = app.use_app('pyside')
QtCore = app_object.backend_module.QtCore,
QtGui = app_object.backend_module.QtGui


vertex = """
#version 120
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float linewidth;
attribute vec4  fg_color;
attribute vec4  bg_color;
attribute float radius;
attribute vec3  position;
varying float v_pointsize;
varying float v_radius;
varying vec4  v_fg_color;
varying vec4  v_bg_color;
varying vec4 mvp;
void main (void)
{
    v_radius = radius;
    v_fg_color = fg_color;
    v_bg_color = bg_color;
    mvp = projection * view * model * vec4(position,1.0);
    gl_Position = mvp;
    gl_PointSize = 2 * (v_radius + linewidth);
}
"""

vertex2 ="""
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
attribute vec3 position;
varying vec4 mvp;
void main()
{
    mvp = projection * view * model * vec4(position,1.0);
    gl_Position = mvp;
}
"""

fragment = """
#version 120
uniform float linewidth;
varying float v_radius;
varying vec4  v_fg_color;
varying vec4  v_bg_color;

void main()
{
    float r = (v_radius + linewidth );
    //is fragment within circle or no?
    float signed_distance = length(gl_PointCoord.xy - vec2(0.5,0.5)) * 2 * r - v_radius;

    gl_FragColor = v_bg_color;
    if( signed_distance >= 0 ) {
        //ring around circles
        if( abs(signed_distance) < (linewidth) ) {
            gl_FragColor = v_fg_color;
        } else {
            discard;
        }
    }
}
"""

fragment2 = """
void main()
{
    gl_FragColor = vec4(0.635, 0.984, 0.756, .5);
}
"""



class Canvas(app.Canvas):

    def __init__(self, n, parent,**kwargs):
        app.Canvas.__init__(self, size=(400, 400),**kwargs)
        #self.native.setLayout(QtGui.QLayout())

        #bind shaders to programs
        self.points = gloo.Program(vertex, fragment)
        self.lines = gloo.Program(vertex2,fragment2)

        # Set attributes
        positions = self.vogel_sphere(n)
        index = np.zeros((n, 2), dtype=np.uint32)
        index[:, 0] = n
        index[:, 1] = np.arange(n, dtype=np.uint32)

        #upload data to shaders
        shader_pos = positions[:,:3]
        print 'initial positions', shader_pos[:n,:]
        self.points['position'] = gloo.VertexBuffer(shader_pos[:n,:])
        self.lines['position'] = gloo.VertexBuffer(shader_pos)

        self.points['radius'] = 5
        self.points['fg_color'] = 0.984, 0.980, 0.635, .5
        colors = np.random.uniform(0.75, 1.00, (n, 4)).astype(dtype=np.float32)
        colors[:, 3] = 1
        self.points['bg_color'] = colors
        self.points['linewidth'] = 1.0

        #set up matrices
        self.view=translate((0, 0, -5.0))
        self.model = np.eye(4, dtype=np.float32)
        self.projection = ortho(-1.5, 1.5, -1.75, 1.75, 0.1, 100)
        #self.projection = perspective(45.0, 328/400.0, 0.1, 100.0)
        self.mvp = np.dot(self.view, self.projection)

        #bind matrices
        self.update_projections()
        self.update_models()
        self.update_views()

        #bind variables
        self.index = gloo.IndexBuffer(index)
        self.positions = positions[:n,:]
        self.world = np.ndarray((n,4))
        self.world[:,:] = self.positions[:,:]
        self.tree = Geometry.kdtree(self.positions.tolist())
        self.colors = colors
        self.theta = 0
        self.phi = 0
        self.oldx, self.oldy = 0,0
        self.radius = self.points['radius']+self.points['linewidth']

        ##dealing with hovering/selecting intricacies
        self.prev_data = np.ndarray((1,5))
        self.prev_data[0,0] = 0
        self.prev_data[0,1:] = colors[0,:]
        self.isPrev = False
        self.isSelect = -1
        self.isPressed = False
        self.isDragged = False
        self.isHover = False
        self.num = n

        #calculate unprojected radius for ray casting
        self.viewport = (0,0,328,400) #setup fake viewport
        p1 = self.unProject(0,10,10)
        p2 = self.unProject(self.radius,10,10)
        self.radius_unproject = p2[0]-p1[0]

        #bind mouse callbacks
        self.events.mouse_press.connect(self.on_mouse_press)
        self.events.mouse_release.connect(self.on_mouse_release)
        self.events.mouse_move.connect(self.on_mouse_move)

        #init gl
        gloo.set_clear_color((1, 1, 1, 1))
        gloo.set_state(depth_test=True)

        #self._timer = app.Timer('auto', connect=self.update_transforms)
        #self._timer.start()

        # Label
        self.label = QtGui.QLabel("Point Name", self.native)
        #self.label.setGeometry(100, 100, 75, 20)
        self.label.setStyleSheet("background-color: white;")
        self.label.hide()

    def vogel_sphere(self,n):
        """
        equally distributes points on sphere using vogel's algorithm
        :param n:
        :return:
        """
        golden_angle = np.pi * (3 - np.sqrt(5))
        theta = golden_angle * np.arange(n)
        y = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
        radius = np.sqrt(1 - y * y)
        #radius = np.random.uniform(0,1,n)

        points = np.zeros((n + 1, 4), dtype=np.float32)
        points[:n, 0] = radius * np.sin(theta)
        points[:n, 1] = y
        points[:n, 2] = radius * np.cos(theta)
        points[n, :] = 0

        #indices
        points[:,3]= np.arange(n+1)

        return points



    def on_resize(self, event):
        width = self.physical_size[0]
        height = self.physical_size[1]
        print 'resize', width, height
        gloo.set_viewport(0, 0, width, height)
        self.viewport = (0, 0, width, height)

    def on_draw(self, event):
        """
        Actually drawing
        """
        gloo.clear()
        self.points.draw('points')
        self.lines.draw('lines', self.index)

    def on_mouse_press(self,event):
        self.oldx, self.oldy = event.pos[0], event.pos[1]
        self.isPressed = True
        self.isHover = False

    def on_mouse_release(self, event):

        #if user done rotating
        if self.isDragged:
            self.model = np.dot(rotate(self.theta, (0, 1, 0)),rotate(self.phi,(1, 0, 0))).astype(np.float32)
            self.update_models()

            # update world positions of points
            self.world = np.ndarray((self.num, 4), dtype=np.float32)
            self.world[:, :3] = self.positions[:, :3]
            self.world[:, 3] = 1
            self.world = np.dot(self.world, self.model)

            self.world[:, 3] = self.positions[:, 3]

            # reset kd tree
            self.tree = Geometry.kdtree(self.world.tolist())

        #user just clicked
        else:
            print 'mouse picking'
            hit = self.ray_cast(event.pos[0], event.pos[1])
            self.restorePrev()
            if hit>-1:
                #deselection
                if hit == self.isSelect:
                    print 'deselecting chosen'
                    self.label.hide()
                    print 'label hide'
                    self.isSelect = -1
                    self.isPrev = False
                else:
                    print 'selecting'
                    self.isSelect = hit
                    self.switchPrev(hit)

                self.update_colors()
                self.update()

        # handle label
        if self.isSelect > -1:
            self.handleLabel(self.isSelect)

        self.isPressed = False
        self.isDragged = False
        self.update()

    def on_mouse_move(self, event):
        #if mouse dragging
        if self.isPressed:
            self.label.hide()
            print 'label hide'
            self.isDragged = True
            newx, newy = event.pos[0],event.pos[1]
            dx, dy = (newx-self.oldx)/5., (newy-self.oldy)/5.

            self.theta+=dx
            self.phi+=dy
            self.model=np.dot(rotate(self.theta, (0, 1, 0)), rotate(self.phi, (1, 0, 0)))

            self.update_models()

            self.oldx = newx
            self.oldy = newy

            self.update()

        #mouse hovering
        else:
            self.isHover = True
            #print 'hover'

            # check if we even need to search intersections
            # we dont care if just hovering with an item already selected
            if self.isSelect<0:
                hit = self.ray_cast(event.pos[0], event.pos[1])
                if hit>-1:
                    # move label
                    self.handleLabel(hit)

                    #highlight point
                    self.restorePrev()
                    self.switchPrev(hit)
                    self.update_colors()
                    self.update()

                elif self.isPrev:
                    #if not hit clear prev buffer
                    #print 'no highlight'
                    self.label.hide()
                    self.restorePrev()
                    self.isPrev = False
                    self.update_colors()
                    self.update()


    def ray_cast(self,screenx,screeny):

        #convert from windows screen coords to opengl screen coords
        screeny = self.viewport[3]-screeny
        #reverse modelview and projection
#        test_front = unProject(tvec3(screenx,screeny,0),np.matrix(self.projection),np.matrix(mv),viewport)
#        test_back = unProject(tvec3(screenx,screeny,1),np.matrix(self.projection),np.matrix(mv),viewport)

        #convert mouse screen coords to world screen coords on near and far clipping plane
        world_x, world_y, world_z_high = self.unProject(screenx,screeny,0)
        world_z_low = self.unProject(screenx,screeny,1)[2]

        #only calc points that give unique bounds
        low_x = world_x - self.radius_unproject
        high_x = world_x + self.radius_unproject

        low_y = world_y - self.radius_unproject
        high_y = world_y + self.radius_unproject

        """
        #brute force
        row = self.positions.shape[0]
        hits = []
        sorted = self.world[self.world[:,2].argsort()]
        #print 'sorted', sorted[:5,:]
        for r in range(row-1,-1,-1):
            x,y,z,index = sorted[r,:]
            if low_x<=x<=high_x:
                if low_y<=y<=high_y:
                    if world_z_low<=z<=world_z_high:
                        return int(index)
        return -1
        """

        #find points that fall inside query volume produced from mouse click
        hits = Geometry.range_query(self.tree,[(low_x,high_x),(low_y,high_y),(world_z_low,world_z_high)])

        #check for intersections
        if len(hits)>0:
            try:
                hits = sorted(hits, key = lambda x: self.world[x,2]*-1)
                return hits[0]
            except:
                raise
        return -1

    def switchPrev(self,index):
        """
        set new previously selected for future deselection
        :param index:
        :return:
        """
        self.prev_data[0,0] = index
        self.prev_data[0,1:] = self.colors[index,:]
        self.isPrev = True
        self.setHighlight(index)

    def setHighlight(self,index, col =(0.823529, 0.411765, 0.117647, 1)):
        self.colors[index, :] = col

    def restorePrev(self):
        """
        switch previously selected colors back to original colors
        :return:
        """
        prev_index = int(self.prev_data[0, 0])
        prev_color = self.prev_data[0, 1:]
        self.colors[prev_index, :] = prev_color

    def update_models(self):
        self.points['model'] = self.model
        self.lines['model'] = self.model

    def update_projections(self):
        self.points['projection'] = self.projection
        self.lines['projection'] = self.projection

    def update_views(self):
        self.points['view'] = self.view
        self.lines['view'] = self.view

    def update_colors(self):
        self.points['bg_color'] = self.colors

    def update_transforms(self, event):
        self.theta += .5
        self.phi += .5

        theta = self.theta
        phi = self.phi

        self.model = np.eye(4, dtype=np.float32)
        rotate(self.model, theta, 0, 0, 1)
        rotate(self.model, phi, 0, 1, 0)
        self.update_models()

        self.update()

    def unProject(self,x,y,z):
        """
        convert screen coords to worldspace coords
        z=0 for near clipping plane
        z=1 for far clipping plane
        :param x:
        :param y:
        :param z:
        :return numpy array:
        """
        inverse = np.linalg.inv(self.mvp)

        win4 = np.array((x,y,z,1.), dtype= np.float32)

        win4[0]=(win4[0] - self.viewport[0]) / float(self.viewport[2])
        win4[1]=(win4[1] - self.viewport[1]) / float(self.viewport[3])
        win4 = win4 * 2. - 1.

        world_coords = np.dot(win4,inverse)
        world_coords/=world_coords[3]

        return world_coords[:3]

    def project(self,x,y,z):
        """
        convert worldspace coords to screen coords
        :param x:
        :param y:
        :param z:
        :return numpy array:
        """
        win4 = np.array((x,y,z,1.), dtype= np.float32)
        win4 = np.dot(win4,self.mvp)
        """
        win4 = np.dot(win4,self.model)
        win4 = np.dot(win4, self.view)
        win4 = np.dot(win4,self.projection)
        """

        win4/=win4[3]
        win4 = (win4+1)/2.0
        win4[0] = win4[0] * (self.viewport[2]) + (self.viewport[0])
        win4[1] = self.viewport[3] - win4[1] * (self.viewport[3]) + (self.viewport[1])

        return win4[:3]

    def handleLabel(self,index):
        x, y, z = self.world[index, :3]
        if z >= -.005:
            x, y, z = self.project(x, y, z)
            self.label.move(x + self.radius, y)
            self.label.setText(str(index))
            width = self.label.fontMetrics().boundingRect(self.label.text()).width()
            self.label.setFixedWidth(width)
            self.label.show()

class TextField(QtGui.QPlainTextEdit):

    def __init__(self, parent):
        QtGui.QPlainTextEdit.__init__(self, parent)
        # Set font to monospaced (TypeWriter)
        font = QtGui.QFont('')
        font.setStyleHint(font.TypeWriter, font.PreferDefault)
        font.setPointSize(8)
        self.setFont(font)


class MainWindow(QtGui.QWidget):

    def __init__(self):
        QtGui.QWidget.__init__(self, None)

        self.setMinimumSize(600, 400)

        # Create a label
        self.vertLabel = QtGui.QLabel("Vertex code", self)

        # Create two editors
        self.vertEdit = TextField(self)
        self.vertEdit.setPlainText(vertex)


        # Create a canvas
        self.canvas = Canvas(500,parent=self)

        # Layout
        hlayout = QtGui.QHBoxLayout(self)
        self.setLayout(hlayout)
        vlayout = QtGui.QVBoxLayout()
        #
        hlayout.addLayout(vlayout, 1)
        hlayout.addWidget(self.canvas.native, 1)
        #
        vlayout.addWidget(self.vertLabel, 0)
        vlayout.addWidget(self.vertEdit, 1)


        self.show()

    def on_compile(self):
        vert_code = str(self.vertEdit.toPlainText())
        frag_code = str(self.fragEdit.toPlainText())
        #self.canvas.program.set_shaders(vert_code, frag_code)
        # Note how we do not need to reset our variables, they are
        # re-set automatically (by gloo)


if __name__ == '__main__':
    app.create()
    m = MainWindow()
    app.run()
