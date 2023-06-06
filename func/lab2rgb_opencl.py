#!/usr/bin/env python

import numpy as np
import pyopencl as cl
import time

def prepare_OpenCL_lab2rgb( L_res = None, a_res = None, b_res = None ):
    import skimage.color
    
    if L_res is None: L_res = 256
    if a_res is None: a_res = 256
    if b_res is None: b_res = 256
    
    ## Our precomputed look-up table.
    labcube = np.zeros( (L_res,a_res,b_res,3), dtype = float )
    labcube[:,:,:,0] = np.linspace(0,100,L_res)[:,None,None]
    labcube[:,:,:,1] = np.linspace(-128,127,a_res)[None,:,None]
    labcube[:,:,:,2] = np.linspace(-128,127,b_res)[None,None,:]
    LUT = skimage.color.lab2rgb( labcube ).astype(np.float32)
    
    ## Padding was important last time I wrote OpenCL code previously (`pyopencl_example.py`)
    
    device = 'gpu'
    if device == 'ask':
        ## Ask the user:
        ctx = cl.create_some_context()
    else:
        ## Choose CPU or GPU automatically.
        platform = cl.get_platforms()
        if device == 'gpu':
            my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        if device == 'cpu' or len(my_gpu_devices) == 0:
            my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.CPU)
        if len(my_gpu_devices) == 0:
            raise RuntimeError( "Unknown device: %s" % device )
        print( my_gpu_devices )
        ctx = cl.Context(devices=my_gpu_devices)
    
    queue = cl.CommandQueue(ctx)
    
    mf = cl.mem_flags
    LUT_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=LUT)
    
    prg = cl.Program(ctx, """
        __kernel void lab2rgb(const unsigned int size, __global const float * LUT, __global const float * lab, __global float * rgb) {
        
        int i = get_global_id(0);
        
        // 1. Offset Lab-space to [0,100]x[0,255]x[0,255]
        // 2. Scale by the lookup table resolution
        // 3. Round to the nearest integer, clip to the lookup table array bounds, and convert to integers.
        // 4. Look up the values.
        
        int L_index = clamp( convert_int( ( lab[3*i+0] )*( %(L_res)s - 1 )/100.0f + 0.5f ), 0, %(L_res)s - 1 );
        int a_index = clamp( convert_int( ( lab[3*i+1] + 128.0f )*( %(a_res)s - 1 )/255.0f + 0.5f ), 0, %(a_res)s - 1 );
        int b_index = clamp( convert_int( ( lab[3*i+2] + 128.0f )*( %(b_res)s - 1 )/255.0f + 0.5f ), 0, %(b_res)s - 1 );
        
        // rgb[ 3*i + 0 ] = LUT[ L_index * a_res * b_res * 3 + a_index * b_res * 3 + b_index * 3 + 0 ];
        rgb[ 3*i + 0 ] = LUT[ ( ( L_index * %(a_res)s + a_index ) * %(b_res)s + b_index ) * 3 + 0 ];
        rgb[ 3*i + 1 ] = LUT[ ( ( L_index * %(a_res)s + a_index ) * %(b_res)s + b_index ) * 3 + 1 ];
        rgb[ 3*i + 2 ] = LUT[ ( ( L_index * %(a_res)s + a_index ) * %(b_res)s + b_index ) * 3 + 2 ];
        }
        """ % { 'L_res': L_res, 'a_res': a_res, 'b_res': b_res }).build()
    
    NO_COPY = True
    
    all_times = []
    def actually_lookup( lab_data ):
        input_shape = lab_data.shape
        lab_data = lab_data.reshape(-1,3).astype(np.float32)
        lab_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lab_data )
        output_shape = lab_data.shape
        
        ## Allocate our return value (CPU and GPU buffers)
        rgb_data = np.empty( lab_data.shape, dtype = np.float32 )
        if NO_COPY:
            rgb_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=rgb_data )
        else:
            rgb_buf = cl.Buffer(ctx, mf.WRITE_ONLY, rgb_data.nbytes )
        
        t0 = time.time()
        ## Run the code.
        ## Automatic
        localsize = None
        # print( 'global size:', output_shape )
        if output_shape[0] % 4 == 0: localsize = (4,1)
        # localsize = (2,1)
        # print( 'local size:', localsize )
        event = prg.lab2rgb( queue, (len(lab_data),1), localsize, np.int32(output_shape[0]), LUT_buf, lab_buf, rgb_buf )
        ## Copy the result back.
        if NO_COPY:
            event.wait()
        else:
            cl.enqueue_copy( queue, rgb_data, rgb_buf )
        t1 = time.time()
        delta_t=t1-t0
        all_times.append( delta_t )
        
        # print( final_matrix[:10,:10] )
        
        # print( np.average( np.asarray( all_times ) ) )
        # print( "Latest time:", delta_t )
        
        rgb_data.shape = input_shape
        return rgb_data
    
    def get_times():
        return np.asarray( all_times )
    
    return actually_lookup, get_times

def OpenCL_test( lab ):
    lab2rgb, get_times = prepare_OpenCL_lab2rgb()
    
    for i in range(5):
        data = lab2rgb( lab )
    
    print( data[:10] )
    
    return data, get_times()

if __name__== "__main__":
    
    npix = 1000*1000
    
    np.random.seed(0)
    lab = (np.random.random((npix,3))*np.array([100,255,255]) - np.array([0,128,128])) #.reshape((1000*1000,3))
    rgb, times = OpenCL_test( lab )
    
    print( 'OpenCL Lab2RGB times:' )
    print( times )
    print( 'min:', times.min() )
    print( 'max:', times.max() )
    print( 'average:', np.average( times ) )
