from Point import Point
import math
import pickle
import kdtree as kd




def kdtree(points):
    tree = kd.create(points)
    return tree

def save_tree(tree):
    pickle.dump(tree,"kdtree")

def load_tree():
    tree = pickle.load("kdtree.pkl")
    return tree

def range_query(root,bounds):
    #type: (kd.KDNODE, List[float]) -> any
    hits=[]
    proceed = _validate(root,bounds)
    if proceed:
        _range_query(root,bounds,0,len(bounds),hits)
    return hits

#check bounds
def _validate(root,bounds):
    if len(root.data)-1 != len(bounds):
        print len(root.data), len(bounds)
        raise ValueError('Data dimensions and bound dimensions don\'t match')
    if len(bounds) == 0:
        return False
    return True

def _range_query(node,bounds,dim,total_dim,hits):
    point = node.data
    if point:
        if _inside_bounds(point,bounds):
            hits.append(int(point[total_dim]))
        try:
            lo, hi = bounds[dim]
        except ValueError:
            print 'check dimensions at '+str(dim)
            raise

        left = point[dim]>=lo
        right = point[dim]<=hi

        #check left and right subtrees
        if left:
            _range_query(node.left, bounds, (dim+1)%total_dim, total_dim,hits)
        if right:
            _range_query(node.right, bounds, (dim+1)%total_dim, total_dim,hits)


#check if point is inside query shape
def _inside_bounds(point, bounds):

    for i in xrange(len(bounds)):
        try:
            lo = bounds[i][0]
            hi = bounds[i][1]
            if point[i]<lo or point[i]>hi:
                return False
        except TypeError:
            if type(bounds[i])== int:
                print 'bad dimensions at bound '+str(i)+', need high and low bound'
            raise
        except IndexError:
            print 'check dimensions at bound '+str(i)
            raise

    return True


def square_dist(point1,point2):
    sum = (point1[0]-point2[0])**2+(point1[1]-point2[1])**2
    if len(point1) == 3:
        sum+=(point1[2]-point2[2])**2
    return sum

def intersect_vectors(v,u, radius):
    cross_prod = Point(v.y * u.z - v.z * u.y, -v.x * u.z + v.z * u.x, v.x * u.y - v.y * u.x)
    mag_cross = math.sqrt(cross_prod.x ** 2 + cross_prod.y ** 2 + cross_prod.z ** 2)
    base = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
    height = mag_cross / base
    if (height <= radius): return True
    return False
