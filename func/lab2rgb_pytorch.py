import numpy as np
from skimage import color

import torch
if not torch.backends.mps.is_available(): raise ImportError( "No PyTorch with Metal backend." )

_lab2rgb_LUT_pytorch = None
def lab2rgb_LUT_pytorch( lab ):
    """
    Given:
        lab: An image in CIE Lab format. The last dimension must be 3 (Lab), with L in the range [0,100], ab in the range [-128,127]^2.
    Returns:
        rgb: An image with the same dimensions as `lab` in RGB space.
    """
    
    gpu = torch.device("mps")
    
    L_res = a_res = b_res = 256
    ## Our precomputed look-up table.
    global _lab2rgb_LUT_pytorch
    if _lab2rgb_LUT_pytorch is None:
        ## Create it the first time.
        labcube = np.zeros( (L_res,a_res,b_res,3), dtype = float )
        labcube[:,:,:,0] = np.linspace(0,100,L_res)[:,None,None]
        labcube[:,:,:,1] = np.linspace(-128,127,a_res)[None,:,None]
        labcube[:,:,:,2] = np.linspace(-128,127,b_res)[None,None,:]
        _lab2rgb_LUT_pytorch = torch.from_numpy( color.lab2rgb( labcube ).astype(np.float32) ).to( gpu )
    
    #import pdb
    #pdb.set_trace()
    
    #import time
    #start = time.time()
    
    lab_torch = (
        (
            torch.from_numpy( lab.reshape(-1,3).astype(np.float32) ).to( gpu )
            +
            torch.tensor( (0,128,128), dtype=torch.float32, device=gpu )
        )
        *
        torch.tensor( ((L_res-1)/100,(a_res-1)/255,(b_res-1)/255), dtype=torch.float32, device=gpu )
    ).round_().type(torch.long).to(gpu).clamp_( torch.tensor( (0,0,0), device=gpu ), torch.tensor( (L_res-1,a_res-1,b_res-1), device=gpu ) )
    rgb = _lab2rgb_LUT_pytorch[ tuple( lab_torch.T ) ].numpy( force = True )
    rgb.shape = lab.shape
    
    # print( "faster conversion took:", time.time() - start )
    
    #start = time.time()
    #rgb_slow = color.lab2rgb( lab )
    #print( "skimage conversion took:", time.time() - start )
    
    ## Always ~1/255
    #print( "lab2rgb diff:", np.abs( rgb_slow - rgb ).sum(axis=-1).max() )
    
    return rgb
