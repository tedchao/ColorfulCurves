import numpy as np
from skimage import io, color

from scipy.spatial import ConvexHull
from scipy.interpolate import pchip_interpolate

from time import time

try:
    from .aux.simplify_convexhull import *
    from .aux.weights import *
    from .aux.simplepalettes import *
    
    #from .highlight_shadow_control import *
    from .alternative_optimization import *
    #from .color_optimizer import *
    #from .lum_optimizer_scale import *

except ImportError:
    from aux.simplify_convexhull import *
    from aux.weights import *
    from aux.simplepalettes import *
    
    #from highlight_shadow_control import *
    from alternative_optimization import *
    #from color_optimizer import *
    #from lum_optimizer_scale import *


def extract_AB_palette( img, palette_num ):
    '''
    Given:
        `img`: A h-w-3 image in LAB-space

    Return:
        `ab_palette`: palette in AB-space
    '''
    ab_pts = img[:, :, 1:].reshape( -1, 2 )
    hull = ConvexHull( ab_pts )
    simp_hull = simplify_convexhull( hull, palette_num )
    ab_palette = simp_hull.points[ simp_hull.vertices ]
    ab_palette = np.vstack( ( ab_palette, np.array( [0, 0] ) ) )        # append `black` as palette
    return ab_palette

def get_palette_img( palette, L0, weights ):
    '''
    Given:
        `palette`: A palette-size-by-2 AB-space palette
        `L0`: A array of size h-by-w
        `weights`: A array of size h*w-by-#p
    
    Return:
        `palette_img`: A RGB palette image for visualization
    '''
    # use weighted average of luminances for each palette color
    w = weights.reshape( (L0.shape[0], L0.shape[1], palette.shape[0]) )
    palette_img = np.zeros( ( palette.shape[0], 3 ) )
    for i in range( palette.shape[0] ):
        palette_img[i, 0] = 100 * (np.sum( np.multiply(w[:, :, i], L0) ) / np.sum( w[:, :, i] ))
    palette_img[:, 1:] = palette
    
    palette_img = color.lab2rgb( palette_img )
    palette_img = np.ascontiguousarray( palette2swatch( palette_img ).transpose( ( 1, 0, 2 ) ) )
    return palette_img

def extract_weights( palette, img ):
    '''
    Given:
        `palette`: A palette-size-by-2 array 
        `img`: A h-w-3 image in LAB-space

    Return:
        `weights`: A N-by-palette-size array
    '''
    h, w = img.shape[0], img.shape[1]
    
    # projected to ab convexhull to avoid negative weights
    ab_pts = project_onto_ab_hull( palette, img[:, :, 1:].reshape( -1, 2 ) ).reshape( ( h, w, 2 ) )
    weights = ABXY_weights( palette, ab_pts )
    return weights
    
def get_recon_img_numpy( palette, weights, luminance ):
    '''
    Given: 
        `palette`: A #P-by-2 array 
        `weights`: A N-by-#P array
        `luminance`: A h-w-1 array of luminance
    
    Return:
        `recon_img`: A reconstructed LAB image of size h-w-3 
    '''
    h, w = luminance.shape[0], luminance.shape[1]
    recon_ab = ( weights @ palette ).reshape( ( h, w, 2 ) )
    recon_img = np.zeros( ( h, w, 3 ) )
    recon_img[:, :, 0] = luminance
    recon_img[:, :, 1:] = recon_ab
    return recon_img

def get_recon_img_pytorch( palette, weights, luminance ):
    '''
    Given: 
        `palette`: A #P-by-2 array 
        `weights`: A N-by-#P array
        `luminance`: A h-w-1 array of luminance
    
    Return:
        `recon_img`: A reconstructed LAB image of size h-w-3 
    '''
    h, w = luminance.shape[0], luminance.shape[1]
    
    import torch
    gpu = torch.device("mps")
    
    palette = torch.from_numpy( palette.astype(np.float32) ).to( gpu )
    weights = torch.from_numpy( weights.astype(np.float32) ).to( gpu )
    
    recon_ab = ( weights @ palette ).numpy( force = True ).reshape( ( h, w, 2 ) )
    recon_img = np.zeros( ( h, w, 3 ) )
    recon_img[:, :, 0] = luminance
    recon_img[:, :, 1:] = recon_ab
    return recon_img

def get_recon_img( *args ):
    import time
    start = time.time()
    n = get_recon_img_numpy( *args )
    print( "NumPy:", time.time() - start )
    start = time.time()
    p = get_recon_img_pytorch( *args )
    print( "PyTorch:", time.time() - start )
    print( "Max Abs Error:", np.abs( n - p ).max() )
    return p

## NumPy is 1.5x the speed of PyTorch.
## It looks like the PyTorch spends a lot of time copying weights to the GPU (which
## really only needs to be done once). But py-spy makes it look like the pytorch
## version is much faster, yet time.time() disagrees. Perhaps py-spy doesn't account
## for some GPU time?
get_recon_img = get_recon_img_numpy

def compute_new_luminance_less_grey_weights( weights ):
    '''
    Given:
        `weights`: A N-by-#P array of weights
    
    Returns:
        `lum_weights`: weights with less grey
    '''
    
    # ignore weights for grey and re-normalize it
    lum_weights = np.copy( weights )
    lum_weights[:, -1] /= 100
    lum_weights = (lum_weights.T / np.sum(lum_weights, axis=1)).T
    return lum_weights

def compute_new_luminance_fast( L0, L_bilap, lum_weights ):
    '''
    Given:
        `L0`: A N-by-M array of original luminance
        `L_bilap`: A N-by-#P array of bilaplacian solutions
        `lum_weights`: A N-by-#P array of weights with less grey as returned by `compute_new_luminance_less_grey_weights()`
    
    Return:
        Reconstructed Luminance
    '''
    
    L_new = 0
    
    L0_ = L0.reshape( -1 )
    
    ## We'll do a fast interpolation ourselves.
    ## Multiply floating point values by the largest index.
    L0_index = ( L0_ * L_bilap.shape[0]-1 ).clip(0,L_bilap.shape[0]-1)
    ## Get the floor, ceiling, and remainder.
    L0_rem, L0_floor = np.modf( L0_index )
    L0_floor = L0_floor.astype(int)
    L0_ceil = np.ceil( L0_index ).astype(int)
    
    for i in range( lum_weights.shape[1] ):
        
        ## Linearly interpolate between the floor and ceiling indices
        ## based on the remainder.
        L0_bilap_floor = L_bilap[L0_floor,i]
        f_L0_ = L0_bilap_floor + L0_rem*( L_bilap[L0_ceil,i] - L0_bilap_floor )
        
        L_new += np.multiply( f_L0_, lum_weights[:, i] )
    return L_new.reshape( L0.shape )

def compute_new_luminance( L0, L_bilap, weights ):
    '''
    Given:
        `L0`: A N-by-M array of original luminance
        `L_bilap`: A N-by-#P array of bilaplacian solutions
        `weights`: A N-by-#P array of weights
    
    Return:
        Reconstructed Luminance
    '''
    
    # ignore weights for grey and re-normalize it
    lum_weights = compute_new_luminance_less_grey_weights( weights )
    return compute_new_luminance_fast( L0, L_bilap, lum_weights )
    
    L_new = 0
    N = 100
    L = np.linspace( 0, 1, N )
    L0_ = L0.reshape( -1 )
    for i in range( lum_weights.shape[1] ):
        f_L0_ = np.interp( L0_, L, L_bilap[:,i] )
        L_new += np.multiply( f_L0_, lum_weights[:, i] )
    
    L_new = L_new.reshape( L0.shape )
    
    # L_new_fast = compute_new_luminance_fast( L0, L_bilap, lum_weights )
    # print( "compute_new_luminance() slow vs fast:", np.abs( L_new - L_new_fast ).max() )
    
    return L_new

def optimization_pixel_constraints( color_palette, palette_constraint, curve_constraint, constraints_weight, color_targets, L_cons ):
    '''
    Given:
        `color_palette`: A #P-by-2 array of color palette in ab-space
        `constraints_weight`: A #C-by-#P array of weights at the constraints
        `color_targets`: A #C-by-2 array of targets colors (only AB)
        `L_cons`: A #C-by-2 array of luminance constraints
    
    Return:
        `palette_opt`: A #P-by-2 array of optimized color palette in ab-space
        `L_bilap`: A N-by-#P array of bilaplacian solutions
    '''
    
    itertimes = []
    now = time()
    start = now
    
    constraints_weight = np.array( constraints_weight )
    color_targets = np.array( color_targets )
    L_cons = np.array( L_cons ) / 100
    
    # optimization color sparsity with palette constraints
    palette_target = []
    for pc in palette_constraint:
        palette_target.append( [pc[1], pc[-1]] )

    #palette_opt = color_optimization( color_palette, constraints_weight, color_targets, palette_target )
    
    # setting up number of samples for computing bilaplacian
    N, w_c = 100, 100
    pixel_lum_cons, curve_lum_cons, lum_weights = convert_constraints_and_weights_data( L_cons, curve_constraint, constraints_weight.T, N, color_palette.shape[0] )
    
    lum_scale = np.ones( color_palette.shape[0] )      # this means in LtBtBL
    del_palette = np.zeros_like( color_palette )
    
    L_bilap = linear_sys_solve( pixel_lum_cons, curve_lum_cons, lum_weights, N, w_c, lum_scale, del_palette )
    lum_scale_new = luminance_bilap_scale( L_bilap, N )
    del_palette_new = color_optimization( color_palette, constraints_weight, color_targets, palette_target, lum_scale_new )
    
    itertimes.append( time() - now )
    print( "1st iteration took:", itertimes[-1] )
    now = time()
    
    # convergence criterion
    iterations = 0
    while True:
        iterations += 1
        ldiff = np.linalg.norm( lum_scale - lum_scale_new )
        pdiff = np.linalg.norm( del_palette - del_palette_new )
        print( f"After {iterations} iteration{'' if iterations == 0 else 's'}:" )
        print( "luminance difference:", ldiff )
        print( "palette difference:", pdiff )
        itertimes.append( time() - now )
        print( "iteration took:", itertimes[-1] )
        now = time()
        if ldiff <= 1e-4 and pdiff <= 1:
            print( "Converged" )
            break
        if iterations > 50:
            print( "Terminating without convergence." )
            break
        
        lum_scale = lum_scale_new
        del_palette = del_palette_new
        
        # update luminance
        L_bilap = linear_sys_solve( pixel_lum_cons, curve_lum_cons, lum_weights, N, w_c, lum_scale, del_palette )
        lum_scale_new = luminance_bilap_scale( L_bilap, N )
        
        # update palette
        del_palette_new = color_optimization( color_palette, constraints_weight, color_targets, palette_target, lum_scale_new )
    
    print( "Total time:", time() - start )
    
    palette_opt = del_palette_new + color_palette
    return palette_opt, L_bilap

def curve_function_plot( L_cons, control_points ):
    L = np.linspace( 0, 1, 500 )
    L_range = pchip_interpolate( control_points[:,0], control_points[:,1], L )
    return L, L_range

_lab2rgb_LUT_numpy = None
def lab2rgb_LUT_numpy( lab ):
    """
    Given:
        lab: An image in CIE Lab format. The last dimension must be 3 (Lab), with L in the range [0,100], ab in the range [-128,127]^2.
    Returns:
        rgb: An image with the same dimensions as `lab` in RGB space.
    """
    
    # start = time.time()
    L_res = a_res = b_res = 256
    
    ## Our precomputed look-up table.
    global _lab2rgb_LUT_numpy
    if _lab2rgb_LUT_numpy is None:
        ## Create it the first time.
        labcube = np.zeros( (L_res,a_res,b_res,3), dtype = float )
        labcube[:,:,:,0] = np.linspace(0,100,L_res)[:,None,None]
        labcube[:,:,:,1] = np.linspace(-128,127,a_res)[None,:,None]
        labcube[:,:,:,2] = np.linspace(-128,127,b_res)[None,None,:]
        _lab2rgb_LUT_numpy = color.lab2rgb( labcube )
    
    ## 1. Flatten lab into a #pixel-by-3 array
    #lab_indices = lab.copy().reshape(-1,3)
    ## 2. Offset Lab-space to [0,100]x[0,255]x[0,255]
    #lab_indices += np.array((0,128,128))[...,:]
    ## 3. Scale by the lookup table resolution
    #lab_indices *= np.array(((L_res-1)/100,(a_res-1)/255,(b_res-1)/255))[...,:]
    ## 4. Round to the nearest integer, clip to the lookup table array bounds, and convert to integers.
    #lab_indices += 0.5
    #lab_indices = lab_indices.astype(int)
    #np.clip( lab_indices, 0, (L_res-1,a_res-1,b_res-1), out = lab_indices )
    ## 5. Look up the values.
    #rgb = _lab2rgb_LUT_numpy[ tuple( lab_indices.T ) ]
    ## 6. Restore the input shape.
    
    #print( "fast conversion 1 took:", time.time() - start )
    #start = time.time()
    
    ## As a one-liner (a touch faster for some reason):
    rgb = _lab2rgb_LUT_numpy[ tuple( ( ( lab.reshape( -1, 3 ) + np.array((0,128,128))[...,:] )*np.array(((L_res-1)/100,(a_res-1)/255,(b_res-1)/255))[...,:] ).round().astype(int).clip(0,(L_res-1,a_res-1,b_res-1)).T ) ]
    #print( "fast conversion 2 took:", time.time() - start )
    
    rgb.shape = lab.shape
    
    #print( "fast conversion took:", time.time() - start )
    
    #start = time.time()
    #rgb_slow = color.lab2rgb( lab )
    #print( "skimage conversion took:", time.time() - start )
    
    ## Always ~1/255
    #print( "lab2rgb diff:", np.abs( rgb_pytorch - rgb ).sum(axis=-1).max() )
    
    return rgb


lab2rgb_LUT_pytorch = lab2rgb_LUT_numpy
try:
    from . import lab2rgb_pytorch
    lab2rgb_LUT_pytorch = lab2rgb_pytorch.lab2rgb_LUT_pytorch
    
    ## PyTorch is faster than NumPy. See `performance_lab2rgb.py`
    print( "Using PyTorch lab2rgb" )
    lab2rgb_fast = lab2rgb_LUT_pytorch
except ImportError:
    print( "No PyTorch with Metal backend found. Try one of:" )
    print( "conda install pytorch torchvision torchaudio -c pytorch-nightly" )
    print( "pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu" )


lab2rgb_LUT_opencl = lab2rgb_LUT_numpy
try:
    from . import lab2rgb_opencl
    lab2rgb_LUT_opencl = lab2rgb_opencl.prepare_OpenCL_lab2rgb()[0]
    
    ## PyOpenCL is faster than PyTorch. See `performance_lab2rgb.py`
    print( "Using OpenCL lab2rgb" )
    lab2rgb_fast = lab2rgb_LUT_opencl
except ImportError:
    print( "No PyOpenCL. Try:" )
    print( "pip install pyopencl" )
