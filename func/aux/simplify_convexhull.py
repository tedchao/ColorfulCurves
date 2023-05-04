import numpy as np
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def edge_collapse( hull_vertices ):
    '''
        Given:
            `hull_vertices`: convexhull vertices of given data
        
        Return:
            `new_hull`: a new convexhull with minimum area after an edge collapse
    '''
    
    def get_intersect( a1, a2, b1, b2 ):
        # reference: https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
        '''
        Given:
            `a1`: a point of the first line
            `a2`: another point of the first line
            `b1`: a point of the second line
            `b2`: another point of the second line
            
            Note: `a2` and `b2` is one of edges of convexhull
        
        Return:
            `intersect`: intersection point of vec(a1, a2) and vec(b1, b2)
            `dir`: a boolean indicating whether the intersection is in correct direction
        '''
        s = np.vstack( [ a1, a2, b1, b2 ] )
        h = np.hstack( ( s, np.ones((4, 1)) ) ) 
        l1 = np.cross( h[0], h[1] )           # get first line
        l2 = np.cross( h[2], h[3] )           # get second line
        x, y, z = np.cross( l1, l2 )          # point of intersection
        return np.array( [ x/z, y/z ] )
    
    aug_vertices= np.vstack( ( hull_vertices[-1, :], hull_vertices, hull_vertices[:2, :] ) )
    
    areas = []
    new_hulls = []
    for i in range( 1, aug_vertices.shape[0]-2 ):
        a2, b2 = aug_vertices[i], aug_vertices[i+1]     # candidate edge
        a1, b1 = aug_vertices[i-1], aug_vertices[i+2]   # neighbors
        intersect = get_intersect( a1, a2, b1, b2 ) # find intersection of edge collapse
        
        # append new convexhull and corresponding area
        new_hull = ConvexHull( np.vstack( ( hull_vertices, intersect ) ) )
        new_hulls.append( new_hull )
        areas.append( new_hull.volume )
    
    return new_hulls[ areas.index( min( areas ) ) ]

def simplify_convexhull( hull, palette_size ):
    '''
    Given:
        `hull`: a convexhull
        `palette_size`: user-specified palette size
    
    Return:
        `hull`: a simplified convexhull with number of palette-size vertices
    '''
    print( 'Original vertices: ', hull.vertices.shape[0] )
    print( 'Simplifying convexhull...' )
    for i in range( hull.vertices.shape[0] - palette_size ):
        hull = edge_collapse( hull.points[ hull.vertices ] )
    print( 'Done. Final vertices: ', hull.vertices.shape[0] )
    return hull