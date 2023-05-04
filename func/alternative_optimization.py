from skimage import io, color
from PIL import Image

import numpy as np
import cv2

from scipy.interpolate import pchip_interpolate
from scipy.optimize import minimize
from scipy.optimize import Bounds
import scipy.sparse

try:
    from utils import *
except ImportError:
    from .utils import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

###
### Palette optimization
### 
def color_optimization( palette, weights, target_color, palette_cons, lum_scale, verbose=False ):
    '''
    Given:
        `palette`: A #p-by-2 array
        `weights`: A array of shape (C, #p)
        `target_colors`: A array of shape (C, 2)
        `palette_cons`: A list of palette constraints. Like: [ [j1, (a1,b1)], [j2, (a2,b2)], ... ]
        `verbose`: toggle for displaying optimization results
    
    Return:
        `palette_opt`: A #p-by-2 array from optimization results
    '''
    np.set_printoptions( suppress = True )
    if np.size( weights ) != 0:
        assert palette.shape[0] == weights.shape[1]
        assert palette.shape[1] == target_color.shape[1]
        assert weights.shape[0] == target_color.shape[0]
    else:
        weights = np.zeros( (1,palette.shape[0]) )
        target_color = np.zeros( (1,2) )
        
    # set up palette constraints
    pc = np.zeros( (len(palette_cons), 2) ) 
    ind = np.zeros( len(palette_cons) ).astype( int )
    for i, (p_indx, cons) in enumerate( palette_cons ): 
        pc[i, :] = cons
        ind[i] = p_indx
    
    # penalize grey on sparsity energy to avoid optimizer prefering to choose grey 
    penal_weights = np.eye( palette.shape[0] )
    penal_weights[-1, -1] = 3


    ### objective function
    def func( vec_delp ):
        delp = vec_delp.reshape( palette.shape )
        delp_penal = penal_weights @ delp
        return np.sum( np.sqrt( 0.01 * np.linalg.norm( delp_penal, axis = 1 ) ** 2 + lum_scale ) )     # optimize for palette's sparsity
    
    ### image-space constraint
    def color_constraints( vec_delp ):
        delp = vec_delp.reshape( palette.shape )
        rec_colors = weights @ ( palette + delp )
        return 1 - np.linalg.norm( rec_colors - target_color, axis = 1 )
    
    ### constraints on palette color
    def palette_constraints( vec_delp ):
        delp = vec_delp.reshape( palette.shape )
        new_palette = palette + delp
        return 1 - np.linalg.norm( new_palette[ ind ] - pc, axis = 1 )
    
    bounds = Bounds( -128 - palette.flatten(), 127 - palette.flatten() )
    
    if len( palette_cons ) == 0:
        cons = [ {'type': 'ineq', 'fun': color_constraints} ]
    else:
        # use inequality on color constraints to avoid singularity
        # (in practice/numerically, it's ok since our sparsity energy is going to dominate this.)
        cons = [ {'type': 'ineq', 'fun': color_constraints}, {'type': 'eq', 'fun': palette_constraints} ]
    
    x0 = np.zeros( palette.shape[0] * palette.shape[1] )
    res = minimize( func, x0, method='SLSQP', constraints=cons, options={'ftol': 1e-6, 'disp': False, 'maxiter': 300}, bounds=bounds )
    delp = ( res.x ).reshape( palette.shape )
    
    if verbose:
        print( '-- Original weights:\n', weights )
        print( '-- Original color:\n', weights @ palette )
        print( '-- Target color:\n', target_color )
        print( '-- Reconstructed color:\n', weights @ ( delp + palette ) )
        
    return delp

###
### Luminance optimization
###
def convert_constraints_and_weights_data( pixel_lum_constraints, curve_lum_constraints, W, N, num_p ):
    pl_cons, cl_cons = [], []
    
    # convert luminance constraints from pixels
    for c in pixel_lum_constraints:
        pl_cons.append( (int(c[0]*N), c[1]) )
        
    # convert luminance constraints from curves themselves
    for c in curve_lum_constraints:
        select_palette = c[0]
        cl_cons.append( (int(c[1]*N) + N*select_palette, c[2]) )
        
    if W.shape[0] != 0:
        W_ = np.copy( W )
        W_[-1, :] /= 100
        W_ = W_ / np.sum(W_, axis=0)
    else:   # if currently no pixel weight constraints, just make it #p-by-1 zero matrix
        W_ = np.zeros( (num_p, 1) ) 
    return pl_cons, cl_cons, W_

def grad_mat( n ):
    G = np.zeros( ( n-1, n ) )
    for i in range( 0, n-1 ):
        G[i,i+1] = 1
        G[i,i] = -1
    return G

_last_bilap = None
def luminance_bilap_scale( lum, N ):
    '''
    Given: 
        `lum`: A N-by-#p array of bilaplacian luminances

    Return:
        `scale`: A #p-array of scale for each luminance curve
    '''
    global _last_bilap
    if _last_bilap is None or _last_bilap.shape[0] != N:
        # building bilaplacian
        G = grad_mat( N )
        Lap = G.T @ G
        _last_bilap = scipy.sparse.csc_matrix( Lap.T @ Lap )
    
    bilap = _last_bilap
    
    lum_scale = np.zeros( lum.shape[1] )
    for i in range( lum.shape[1] ):
        lum_scale[i] = lum[:, i].T @ bilap @ lum[:, i]
    return lum_scale

def linear_sys_solve_slow( constraints, curve_constraints, W, N, w_c, lum_scale, del_palette ):
    '''
    Given:
        `constraints`: A list of luminance constraints, i.e. [ (int, 0.2), (int, 0.3), ... ]
        `curve_constraints`: A list of curve constraints, i.e. [ (int, 0.2), (int, 0.3), ... ]
        `W`: A #p-by-#c array of weights at constraints
        `N`: Number of samples (100 is enough)
        `w_c`: A scalar of the penalization term
    
    Return:
        `L_`: A N-by-#p array of bilaplacian luminances
    '''
    # penalize grey on sparsity energy to avoid optimizer prefering to choose grey 
    penal_weights = np.eye( del_palette.shape[0] )
    penal_weights[-1, -1] = 3
    
    w_sp = 0.01
    
    # identities in kronecker product
    id_p = np.diag( 1 / np.sqrt( lum_scale + w_sp * np.linalg.norm( penal_weights @ del_palette, axis = 1 ) ** 2 ) )
    id_n = np.eye( N ) 
    
    # building bilaplacian
    G = grad_mat( N )
    Lap = G.T @ G
    
    # mirroring the laplacian to avoid prefering 0 gradients at endpoints
    Lap[0,:] = 0
    Lap[-1,:] = 0
    B = Lap
    
    # building selection and constraint matrices
    if len( constraints ) == 0: # if no luminance constraints, then set C and S to be zeros
        C = np.zeros( ( N, 1 ) )
        S = np.zeros( ( N, 1 ) )
    else:
        C = np.zeros( ( N, len( constraints ) ) )
        S = np.zeros( ( N, len( constraints ) ) )
    for i, ( Lindex, val ) in enumerate( constraints ):
        C[Lindex, i] = val
        S[Lindex, i] = 1
        
        
    # compute left-hand and right-hand side
    # note: `flatten('F')` is vectorization on columns
    W_idn_kron = np.kron( W, id_n )
    A = np.kron( id_p, B.T@B ) + w_c * W_idn_kron @ np.diag( (2*np.multiply(S,S)).flatten('F') ) @ np.kron( W.T, id_n )
    b = w_c * ( W_idn_kron @ ( np.multiply( 2*C, S ).flatten('F') ) )
    
    # change the rows to identity for known values
    id_p_n = np.eye( W.shape[0] * N )
    zero_lum_ind = [N*i for i in range(W.shape[0])]
    one_lum_ind = [N*(i+1)-1 for i in range(W.shape[0])]
    
    A[zero_lum_ind, :] = id_p_n[zero_lum_ind, :]
    A[one_lum_ind, :] = id_p_n[one_lum_ind, :]
    b[zero_lum_ind] = 0
    b[one_lum_ind] = 1
    
    # placing curve constraints from direct manipulation of curves
    if len( curve_constraints ) != 0:
        curve_cons_ind = [i[0] for i in curve_constraints]
        curve_cons_val = np.array( [i[1] for i in curve_constraints] )
        
        A[curve_cons_ind, :] = id_p_n[curve_cons_ind, :]
        b[curve_cons_ind] = curve_cons_val
        
    x_l = np.linalg.solve( A, b )
    L_ = x_l.reshape( N, W.shape[0], order='F' )
    
    return L_

## From https://stackoverflow.com/questions/44461658/efficient-kronecker-product-with-identity-matrix-and-regular-matrix-numpy-pyt
def kron_A_I(A, N):
    '''Simulates np.kron(A, np.eye(N))'''
    m,n = A.shape
    out = np.zeros((m,N,n,N),dtype=A.dtype)
    r = np.arange(N)
    out[:,r,:,r] = A
    out.shape = (m*N,n*N)
    return out
def kron_I_A(N, A):
    '''Simulates np.kron(np.eye(N), A)'''
    m,n = A.shape
    out = np.zeros((N,m,N,n),dtype=A.dtype)
    r = np.arange(N)
    out[r,:,r,:] = A
    out.shape = (m*N,n*N)
    return out
def kron_diag_A(d, A):
    '''Simulates np.kron(np.diag(d), A)'''
    m,n = A.shape
    N = len(d)
    out = np.zeros((N,m,N,n),dtype=A.dtype)
    r = np.arange(N)
    out[r,:,r,:] = d[:,None,None] * A
    out.shape = (m*N,n*N)
    return out

## This function is unused but can be very helpful:
def repeated_block_diag_times_matrix( block, matrix ):
    # return scipy.sparse.block_diag( [ block ]*( matrix.shape[0]//block.shape[1] ) ).dot( matrix )
    # print( abs( scipy.sparse.block_diag( [ block ]*( matrix.shape[0]//block.shape[1] ) ).dot( matrix ) - numpy.dot( block, matrix.reshape( block.shape[1], -1, order='F' ) ).reshape( -1, matrix.shape[1], order='F' ) ).max() )
    return np.dot( block, matrix.reshape( block.shape[1], -1, order='F' ) ).reshape( -1, matrix.shape[1], order='F' )

_last_2BTB = None
def linear_sys_solve( constraints, curve_constraints, W, N, w_c, lum_scale, del_palette ):
    '''
    Given:
        `constraints`: A list of luminance constraints, i.e. [ (int, 0.2), (int, 0.3), ... ]
        `curve_constraints`: A list of curve constraints, i.e. [ (int, 0.2), (int, 0.3), ... ]
        `W`: A #p-by-#c array of weights at constraints
        `N`: Number of samples (100 is enough)
        `w_c`: A scalar of the penalization term
    
    Return:
        `L_`: A N-by-#p array of bilaplacian luminances
    '''
    # penalize grey on sparsity energy to avoid optimizer prefering to choose grey 
    penal_weights = np.eye( del_palette.shape[0] )
    penal_weights[-1, -1] = 3
    
    w_sp = 0.001
    
    # identities in kronecker product
    id_p = 1 / np.sqrt( lum_scale + w_sp * np.linalg.norm( penal_weights @ del_palette, axis = 1 ) ** 2 )
    
    global _last_2BTB
    if _last_2BTB is None or _last_2BTB.shape[0] != N:
        # building bilaplacian
        G = grad_mat( N )
        Lap = G.T @ G
        
        # mirroring the laplacian to avoid prefering 0 gradients at endpoints
        Lap[0,:] = 0
        Lap[-1,:] = 0
        B = Lap
        
        _last_2BTB = B.T@B
    
    # compute left-hand and right-hand side
    # note: `flatten('F')` is vectorization on columns
    A = kron_diag_A( id_p, _last_2BTB )
    
    # building selection and constraint matrices
    if len( constraints ) == 0: # if no luminance constraints, then set C and S to be zeros
        C = np.zeros( ( N, 1 ) )
        S = np.zeros( ( N, 1 ) )
        
        b = np.zeros( A.shape[0] )
    else:
        C = np.zeros( ( N, len( constraints ) ) )
        S = np.zeros( ( N, len( constraints ) ) )
        for i, ( Lindex, val ) in enumerate( constraints ):
            C[Lindex, i] = val
            S[Lindex, i] = 1
        
        W_idn_kron = kron_A_I( W, N )
        A += w_c * W_idn_kron @ ( (2*np.multiply(S,S)).flatten('F')[:,None] * kron_A_I( W.T, N ) )
        b = w_c * ( W_idn_kron @ ( np.multiply( 2*C, S ).flatten('F') ) )
    
    # change the rows to identity for known values
    id_p_n = np.eye( W.shape[0] * N )
    zero_lum_ind = [N*i for i in range(W.shape[0])]
    one_lum_ind = [N*(i+1)-1 for i in range(W.shape[0])]
    
    A[zero_lum_ind, :] = id_p_n[zero_lum_ind, :]
    A[one_lum_ind, :] = id_p_n[one_lum_ind, :]
    b[zero_lum_ind] = 0
    b[one_lum_ind] = 1
    
    # placing curve constraints from direct manipulation of curves
    if len( curve_constraints ) != 0:
        curve_cons_ind = [i[0] for i in curve_constraints]
        curve_cons_val = np.array( [i[1] for i in curve_constraints] )
        
        A[curve_cons_ind, :] = id_p_n[curve_cons_ind, :]
        b[curve_cons_ind] = curve_cons_val
        
    #import time
    #start = time.time()
    #x_l = np.linalg.solve( A, b )
    #print( "Dense:", time.time() - start )
    #start = time.time()
    x_l = scipy.sparse.linalg.spsolve( scipy.sparse.csc_matrix( A ), b )
    #print( "Sparse:", time.time() - start )
    ## Sparse is much faster
    L_ = x_l.reshape( N, W.shape[0], order='F' )
    
    # print( 'fast - slow:', np.abs( L_ - linear_sys_solve_slow( constraints, curve_constraints, W, N, w_c, lum_scale, del_palette ) ).max() )
    
    return L_

def compute_new_luminance_local( L0, L_bilap, weights, N ):
    '''
    Given:
        `L0`: A N-by-M array of original luminance
        `L_bilap`: A N-by-#P array of bilaplacian solutions
        `weights`: A N-by-#P array of weights
    
    Return:
        Reconstructed Luminance
    '''
    
    from scipy import interpolate
    
    # ignore weights for grey and re-normalize it
    lum_weights = np.copy( weights )
    lum_weights[:, -1] /= 100
    lum_weights = (lum_weights.T / np.sum(lum_weights, axis=1)).T
    
    L_new = 0
    L = np.linspace( 0, 1, N )
    L0_ = L0.reshape( -1 )
    for i in range( lum_weights.shape[1] ):
        f = interpolate.interp1d( L, L_bilap[:,i] )
        L_new += np.multiply( f( L0_ ), lum_weights[:, i] )
    return L_new.reshape( L0.shape )

def plot_luminance_curves( L_new, N ):
    L = np.linspace( 0, 1, N )
    
    import matplotlib.pyplot as plt
    
    for i in range( L_new.shape[1] ):
        plt.plot( L, L_new[:, i] )
    plt.show()
    
def main():
    import argparse
    parser = argparse.ArgumentParser( description = 'optimization on entire sparsity.' )
    parser.add_argument( 'img', help = 'The path to the input image.' )
    args = parser.parse_args()
    
    img = color.rgb2lab( np.array( Image.open( args.img ) ) / 255. )   
    L0 = img[:, :, 0] / 100
    ab = img[:, :, 1:]
    
    palette_size = 9
    ab_palette = extract_AB_palette( img, palette_size )
    weights = extract_weights( ab_palette, img ).reshape( (img.shape[0], img.shape[1], palette_size+1) )
        
    i, j = 100, 80
    i2, j2 = 400, 650
    print( 'Selected pixel: ',  img[i,j] )
    print( 'Selected pixel: ',  img[i2,j2] )
    target_lum = 0.1
    L_cons = np.array( [[L0[i, j], target_lum]] )
    #L_cons = np.array( [[L0[i, j], target_lum], [L0[i2, j2], 0.4]] )
    constraints_weight = np.array( [ weights[i, j] ] )
    #constraints_weight = np.array( [ weights[i, j], weights[i2, j2] ] )
    
    curve_cons = [ (2, 0.4, 0.1) ]
    #curve_cons = []
    
    N, w_c = 30, 100
    pixel_lum_cons, curve_lum_cons, lum_weights = convert_constraints_and_weights_data( L_cons, curve_cons, constraints_weight.T, N, ab_palette.shape[0] )
    
    color_targets = np.array( [[ 36, 28]] )
    #color_targets = np.array( [[ 10, 13], [ -20, 51]] )
    palette_target = [[1, np.array([  -40, 15 ])]]
    #palette_target = []
    
    # solve linear system multiple times
    import time
    start_time = time.time()
    
    # initialization
    lum_scale = np.ones( ab_palette.shape[0] )      # this means in LtBtBL
    del_palette = np.zeros_like( ab_palette )
    
    L_bilap = linear_sys_solve( pixel_lum_cons, curve_lum_cons, lum_weights, N, w_c, lum_scale, del_palette )
    lum_scale_new = luminance_bilap_scale( L_bilap, N )
    del_palette_new = color_optimization( ab_palette, constraints_weight, color_targets, palette_target, lum_scale_new )
    
    # convergence criterion
    while np.linalg.norm( lum_scale - lum_scale_new ) > 1e-5 or np.linalg.norm( del_palette - del_palette_new ) > 1e-1:
        #print( '---')
        #print( np.linalg.norm( lum_scale - lum_scale_new ) )
        #print( np.linalg.norm( del_palette - del_palette_new ) )
        
        lum_scale = lum_scale_new
        del_palette = del_palette_new
        
        # update luminance
        L_bilap = linear_sys_solve( pixel_lum_cons, curve_lum_cons, lum_weights, N, w_c, lum_scale, del_palette )
        lum_scale_new = luminance_bilap_scale( L_bilap, N )
        
        # update palette
        del_palette_new = color_optimization( ab_palette, constraints_weight, color_targets, palette_target, lum_scale_new )
    print( 'Runtime: ', time.time() - start_time )
    
    plot_luminance_curves( L_bilap, N )
    
    palette_opt = ab_palette + del_palette_new
    
    # reconstruction
    recon_ab = weights @ palette_opt
    recon_lum = compute_new_luminance_local( L0, L_bilap, weights.reshape(-1, ab_palette.shape[0]), N ) * 100
    
    recon_img = np.zeros_like( img )
    recon_img[:, :, 0] = recon_lum
    recon_img[:, :, 1:] = recon_ab
    
    print( 'Reconstruction error: ', np.abs( recon_img[i,j][0] - target_lum*100 ) )
    recon_img = ( color.lab2rgb( recon_img ) * 255. ).clip( 0, 255 ).astype( np.uint8 )
    Image.fromarray( recon_img ).save( '../test-edit-alternative.png' )
    
if __name__ == '__main__':
    main()