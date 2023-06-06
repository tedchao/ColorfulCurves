import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.sparse import coo_matrix

import pyximport
pyximport.install()
from .GteDistPointTriangle import *
import warnings
warnings.filterwarnings("ignore")

def ab_to_abxy( ab ):
    '''
    Given: 
        `ab`: A h-w-2 array in AB-space
    
    Return:
        `abxy`: A h-w-4 array in ABXY-space
    '''
    h, w = ab.shape[0], ab.shape[1]
    x, y = np.meshgrid( np.arange( h ), np.arange( w ) )   # position xy
    abxy = np.zeros( (h, w, 4) )
    abxy[:, :, :2] = ab
    abxy[:, :, 2] = ( 1 / max(h, w) ) * x.T
    abxy[:, :, 3] = ( 1 / max(h, w) ) * y.T
    return abxy

def project_onto_ab_hull( palette, ab_pts ):
    '''
    Given: 
        `palette`: simplified convexhull in AB-space
        `ab_pts`: all AB points
    
    Return:
        `ab_pts`: projected AB points
    '''
    print( 'Project pixels outside onto AB hull...' )
    hull = ConvexHull( palette ) 
    tri = Delaunay( palette )
    simplices = tri.find_simplex( ab_pts, tol=1e-6 )
    
    for i in range( ab_pts.shape[0] ):
        if simplices[i] < 0:
            dist_list=[]
            # compute distance to each face of convexhull
            for j in range( hull.simplices.shape[0] ):
                result = DCPPointTriangle( ab_pts[i], hull.points[hull.simplices[j]] )
                dist_list.append( result )
            sort_dist = sorted( dist_list, key=lambda d: d['distance'] )
            ab_pts[i] = sort_dist[0]['closest']
            
    print( 'Done projection.' )
    return ab_pts

def Star_coordinates( vertices, data ):
    ## Find the star vertex
    star = np.argmin( np.linalg.norm( vertices, axis=1 ) ) # always use grey [0,0] as star vertex
    ## Make a mesh for the palette
    hull = ConvexHull( vertices )
    ## Star tessellate the faces of the convex hull
    simplices = [ [star] + list(face) for face in hull.simplices if star not in face ]
    barycoords = -1*np.ones( ( data.shape[0], len(vertices) ) )
    ## Barycentric coordinates for the data in each simplex
    for s in simplices:
        s0 = vertices[s[:1]]
        b = np.linalg.solve( (vertices[s[1:]]-s0).T, (data-s0).T ).T
        b = np.append( 1-b.sum(axis=1)[:,None], b, axis=1 )
        ## Update barycoords whenever data is inside the current simplex (with threshold).
        mask = (b>=-1e-8).all(axis=1)
        barycoords[mask] = 0.
        barycoords[np.ix_(mask,s)] = b[mask]
    return barycoords

def Delaunay_coordinates( vertices, data ): # Adapted from Gareth Rees 
    '''
    Given:
        `vertices`: convexhull vertices
        `data`: data of interest to compute Delaunay weights
    
    Return:
        Delaunay weights in sparse format
    '''
    tri = Delaunay( vertices )  # Compute Delaunay tessellation. 
    # Find the tetrahedron containing each target (or -1 if not found). 
    simplices = tri.find_simplex( data, tol = 1e-6 ) 
    assert ( simplices != -1 ).all() # data contains outside vertices. 
    # Affine transformation for simplex containing each datum. 
    X = tri.transform[ simplices, :data.shape[1] ] 
    # Offset of each datum from the origin of its simplex. 
    Y = data - tri.transform[ simplices, data.shape[1] ] 
    # Compute the barycentric coordinates of each datum in its simplex. 
    b = np.einsum( '...jk,...k->...j', X, Y ) 
    barycoords = np.c_[ b, 1 - b.sum( axis=1 ) ] 
    # Return the weights as a sparse matrix. 
    rows = np.repeat( np.arange( len( data ) ).reshape( ( -1, 1 ) ), len( tri.simplices[0] ), 1 ).ravel() 
    cols = tri.simplices[ simplices ].ravel() 
    vals = barycoords.ravel() 
    return coo_matrix( ( vals, ( rows, cols ) ), shape=( len( data ), len( vertices ) ) ).tocsr()

def ABXY_weights( palette, ab_pts ):
    '''
    Given:
        `palette`: A palette-size-by-2 array
        `ab_pts`: A h-by-w-by-2 points in AB-space
    
    Return:
        `weights`: A N-by-palette-size array of weights
    '''
    abxy = ab_to_abxy( ab_pts ).reshape( -1, 4 )
    abxy_hull_vertices = abxy[ ConvexHull( abxy ).vertices ]
    w_abxy = Delaunay_coordinates( abxy_hull_vertices, abxy )
    w_ab = Star_coordinates( palette, abxy_hull_vertices[:, :2] )
    return w_abxy.dot( w_ab )