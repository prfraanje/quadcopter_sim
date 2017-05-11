import numpy as np
import trimesh
import pybullet as p
import pyqtgraph.opengl as gl

SPHERE   = 2
BOX      = 3
CYLINDER = 4
MESH     = 5
PLANE    = 6
CAPSULE  = 7

Nres = 50

def bullet2pyqtgraph(Id):
    shape_data = p.getVisualShapeData(Id)
    mesh_items = []
    for i in range(len(shape_data)):
        # process item i
        mesh = visualGeometryType2mesh(shape_data[i])
        mesh_items.append(mesh)
    return mesh_items

def quaternion2axis_angle(quaternion):
    " quaternion: [x,y,z,w] (shape used in Bullet Physics)"
    import numpy.linalg
    if np.isclose(quaternion[3],1):
        x=y=angle=0.
        z=1.
    else:
        a_cos = quaternion[3]
        a_sin = np.linalg.norm(quaternion[0:3])
        #angle = 2 * np.degrees(np.arccos(quaternion[3]))
        angle = 2 * np.arctan2(a_sin,a_cos)
        #scale = np.sqrt(1-quaternion[3]**2)
        scale=a_sin
        x = quaternion[0]/scale
        y = quaternion[1]/scale
        z = quaternion[2]/scale
    return angle,x,y,z

# https://www.astro.rug.nl/software/kapteyn/_downloads/attitude.pdf

def quaternion2roll_pitch_yaw(quaternion):
    """
       quaternion: [x,y,z,w] (shape used in Bullet Physics) to roll, pitch, yaw
       see:
       http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/
    """
    from math import atan2, asin
    qw = quaternion[3]
    qx = quaternion[0]
    qy = quaternion[1]
    qz = quaternion[2]
    # roll = bank, pitch = attitude, yaw = heading
    roll = atan2(2*(qx*qw-qy*qz),1-2*(qx*qx+qz*qz))
    pitch = asin(2*(qx*qy+qz*qw))
    yaw = atan2(2*(qy*qw-qx*qz),1-2*(qy*qy+qz*qz))
    return roll,pitch,yaw

def quaternion2rotation_matrix(quaternion):
    """
       quaternion: [x,y,z,w] (shape used in Bullet Physics) to rotation matrix 
       see:
       http://www.mrelusive.com/publications/papers/SIMD-From-Quaternion-to-Matrix-and-Back.pdf
    """
    import numpy as np
    import math
    # from numpy.linalg import norm
    # quaternion = quaternion/norm(quaternion)
    # q0 = quaternion[3]
    # q1 = quaternion[0]
    # q2 = quaternion[1]
    # q3 = quaternion[2]
    # return np.array([ [1-2*q2*q2-2*q3*q3,2*q1*q2+2*q3*q0,2*q1*q3-2*q2*q0],
    #                   [2*q1*q2-2*q3*q0,1-2*q1*q1-2*q3*q3,2*q2*q3+2*q1*q0],
    #                   [2*q1*q3+2*q2*q0,2*q2*q3-2*q1*q0,1-2*q1*q1-2*q2*q2] ])
    n = np.dot(quaternion,quaternion)
    if n<1e-12:
        return np.identity(3)
    q =np.array([quaternion[3],quaternion[0],quaternion[1],quaternion[2]])
    q *= math.sqrt(2.0/n)
    q = np.outer(q,q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])


# cylinder is a convenience function to create a cylinder shape in
# pyqtgraph/OpenGL, it gives you a number of vertices distributed over the
# surface of the cylinder and triangular shaped faces that cover the whole
# surface of the cylinder
# cylinders are being used to visualize joints
def cylinder_mesh(radius,height,N):
    """Calculates vertices and faces for a cylinder for visualisation in
    pyqtgraph/OpenGL.

    Inputs:
        radius: radius of the cylinder
        height: height of the cylinder
        N: number of segments to approximate the circular shape of the cylinder 

    Outputs:
        vertices: array with on each row the (x,y,z) coordinates of the vertices 
        faces: array with triangular faces of the cylinder

    Note:
        The cylinder is a circle in the x,y plane with center at (0,0) that is
        extruded along the z-axis.

    """
    import numpy as np
    import scipy.spatial
    t = np.linspace(0,2*np.pi,N,endpoint=False).reshape(N,1)
    vertices = np.zeros((2*N,3))
    vertices[0:N,:] = np.hstack((radius*np.cos(t),radius*np.sin(t),np.zeros((N,1))))
    vertices[N:2*N,:] = vertices[0:N,:] + np.hstack((np.zeros((N,2)),height*np.ones((N,1))))
    faces = np.zeros((N-2+2*N+N-2,3),dtype=np.uint)
    # bottom, makes use of Delaunay triangulation contained in Scipy's
    # submodule spatial (which on its turn makes use of the Qhull library)
    faces[0:N-2,:] = scipy.spatial.Delaunay(vertices[0:N,0:2],furthest_site=True,qhull_options='QJ').simplices[:,-1::-1]
    #sides
    for i in range(N-1):
        faces[N-2+2*i,:]   = np.array([i,i+1,N+i+1],dtype=np.uint)
        faces[N-2+2*i+1,:] = np.array([i,N+i+1,N+i],dtype=np.uint)
    # final one between the last and the first:
    faces[N-2+2*(N-1),:]   = np.array([N-1,0,N],dtype=np.uint)
    faces[N-2+2*(N-1)+1,:] = np.array([N-1,N,2*N-1],dtype=np.uint)
    # top
    faces[N-2+2*N:N-2+2*N+N-2,:] = N + faces[0:N-2,-1::-1]

    return vertices,faces

# simular to the cylinder, but now for creating a box-shaped object
# boxes are used to visualize links
def box_mesh(size=(1,1,1)):
    """Calculates vertices and faces for a box for visualisation in
    pyqtgraph/OpenGL.

    Inputs:
        size: 3 element array/list with the width,depth,height, i.e. 
              the dimensions along the x, y and z-axis.

    Outputs:
        vertices: array with on each row the (x,y,z) coordinates of the vertices 
        faces: array with triangular faces of the box 

    Note:
        The box is between (0,0,0) and (size[0],size[1],size[2]), note that
        negative sizes are not prevented but result in strange artifacts because
        it changes the orientation of the faces of the box (inside becomes
        outside).

    """
    import numpy as np
    vertices = np.zeros((8,3))
    faces = np.zeros((12,3),dtype=np.uint)
    xdim = size[0]
    ydim = size[1]
    zdim = size[2]
    vertices[0,:] = np.array([0,ydim,0])
    vertices[1,:] = np.array([xdim,ydim,0])
    vertices[2,:] = np.array([xdim,0,0])
    vertices[3,:] = np.array([0,0,0])
    vertices[4,:] = np.array([0,ydim,zdim])
    vertices[5,:] = np.array([xdim,ydim,zdim])
    vertices[6,:] = np.array([xdim,0,zdim])
    vertices[7,:] = np.array([0,0,zdim])

    faces = np.array([
        # bottom (clockwise, while looking from top)
        [2, 1, 0],
        [3, 2, 0],
        # sides (counter-clock-wise)
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
        # top (counter-clockwise)
        [4, 5, 6],
        [4, 6, 7]
        ],dtype=np.uint)

    return vertices,faces


def visualGeometryType2mesh(shape_data_element):
    def sphere():
        radius = shape_data_element[3][0]
        local_pos = shape_data_element[5]
        local_orient = shape_data_element[6]
        color = shape_data_element[7]
        sphere_data = gl.MeshData.sphere(rows=Nres,cols=Nres,radius=radius)
        mesh = gl.GLMeshItem(meshdata=sphere_data,drawFaces=True,
                    drawEdges=False,smooth=False,computeNormals=True,color=color,shader='shaded',glOptions='opaque')
        angle,x,y,z=quaternion2axis_angle(local_orient)
        mesh.translate(local_pos[0],local_pos[1],local_pos[2],local=False)
        mesh.rotate(np.degrees(angle),x,y,z,local=False)
        return mesh

    def box():
        print('box')
        dims = shape_data_element[3]
        local_pos = shape_data_element[5]
        local_orient = shape_data_element[6]
        color = shape_data_element[7]
        verts,faces=box_mesh(dims)
        verts -= .5*np.array(dims)
        mesh = gl.GLMeshItem(vertexes=verts,faces=faces,drawFaces=True,
                    drawEdges=False,smooth=False,computeNormals=True,color=color,shader='shaded',glOptions='opaque')
        angle,x,y,z=quaternion2axis_angle(local_orient)
        mesh.translate(local_pos[0],local_pos[1],local_pos[2],local=False)
        mesh.rotate(np.degrees(angle),x,y,z,local=False)
        return mesh

    def cylinder():
        print('cylinder')
        radius = shape_data_element[3][1]
        height = shape_data_element[3][0]
        local_pos = shape_data_element[5]
        local_orient = shape_data_element[6]
        color = shape_data_element[7]
        verts,faces=cylinder_mesh(radius,height,Nres)
        verts[:,2] -= .5*height
        mesh = gl.GLMeshItem(vertexes=verts,faces=faces,drawFaces=True,
                    drawEdges=False,smooth=False,computeNormals=True,color=color,shader='shaded',glOptions='opaque')
        angle,x,y,z=quaternion2axis_angle(local_orient)
        mesh.rotate(np.degrees(angle),x,y,z,local=False)
        mesh.translate(local_pos[0],local_pos[1],local_pos[2],local=False)
        return mesh

    def mesh_structure():
        print('mesh')
        scales = shape_data_element[3]
        filename = shape_data_element[4]
        local_pos = shape_data_element[5]
        local_orient = shape_data_element[6]
        color = shape_data_element[7]
        print('color=',color)
        # has to use trimesh here (numpy-stl does not give a trimesh)
        mesh_data = trimesh.load_mesh(shape_data_element[4].decode('utf-8'))
        mesh = gl.GLMeshItem(vertexes=np.array(mesh_data.vertices)*scales,
                faces=np.array(mesh_data.faces),drawFaces=True,
                drawEdges=False,smooth=True,computeNormals=True,color=color,shader='shaded',glOptions='opaque')
        angle,x,y,z=quaternion2axis_angle(local_orient)
        mesh.translate(local_pos[0],local_pos[1],local_pos[2],local=False)
        mesh.rotate(np.degrees(angle),x,y,z,local=False)
        return mesh

    def plane_structure():
        return []

    def capsule_structure():
        return []

    switcher = {
        SPHERE: sphere,
        BOX: box,
        CYLINDER: cylinder,
        MESH: mesh_structure,
        PLANE: plane_structure,
        CAPSULE: capsule_structure
    }
    return switcher.get(shape_data_element[2],lambda :"nothing")()

