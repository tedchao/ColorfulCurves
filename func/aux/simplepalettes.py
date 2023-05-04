#!/usr/bin/env python

from __future__ import print_function, division

from numpy import *
import skimage.io
from skimage import color
import os

## For reproducibility
random.seed(1)

LUMINANCE = 60.
CHROMA = 33.

VERIFY = True

def generate_one_random_lch():
    '''
    Lch has Luminance in the range [0,100],
    chroma in the range [0,100],
    and hue in radians [0,2*pi].
    '''
    
    return ( LUMINANCE, CHROMA, 2*pi*random.random() )

def lch2rgb( lch ):
    '''
    Given a color in Lch color space as a tuple of three numbers,
    returns a color in sRGB color space as a tuple of three numbers.
    '''
    assert len( lch ) == 3
    assert float( lch[0] ) == lch[0]
    assert float( lch[1] ) == lch[1]
    assert float( lch[2] ) == lch[2]
    
    if VERIFY and not lch_color_is_in_sRGB_gamut( lch ):
        print( "Color out of sRGB gamut!" )
    
    return color.lab2rgb( color.lch2lab( asfarray(lch)[None,None,:] ) )[0,0]

def lch_color_is_in_sRGB_gamut( lch ):
    '''
    Given a color in Lch color space as a tuple of three numbers,
    returns True if the color can be converted to sRGB and False otherwise.
    '''
    
    c_lab = color.lch2lab( asfarray(lch)[None,None,:] )
    c_lab2 = color.rgb2lab( color.lab2rgb( c_lab ) )
    return abs( c_lab - c_lab2 ).max() < 1e-5

# def hue_too_close( hue1, hue2 ):
#    return lch_too_close( ( LUMINANCE, CHROMA, hue1 ), ( LUMINANCE, CHROMA, hue2 ) )
def lch_too_close( col1, col2 ):
    ## 2.3 is JND in Lab.
    return linalg.norm( lch2lab(col1) - lch2lab(col2) ) <= 2.3*10
def lch2lab( lch ): return color.lch2lab( asfarray(lch)[None,None,:] )[0,0]

def sort_lch_colors_relative_to_first( seq ):
    seq = asfarray( seq )
    hue0 = seq[0][2]
    for i in range(1,len(seq)):
        while seq[i][2] < hue0: seq[i][2] += 2*pi
    
    seq = seq.tolist()
    seq_sorted = seq[:1] + sorted( seq[1:], key = lambda x: x[2] )
    ## mod the numbers back
    seq_sorted = [ ( c[0], c[1], fmod( c[2], 2*pi ) ) for c in seq_sorted ]
    return seq_sorted

def generate_complementary():
    '''
    Returns two palettes. The first is a complementary palette consisting of
    two RGB colors opposite each other on the Lch color wheel.
    The second palette will share one of the colors, but the other will have random hue.
    All colors will have identical luminance and chroma.
    '''
    
    color1 = generate_one_random_lch()
    color2 = ( color1[0], color1[1], fmod( color1[2] + pi, 2*pi ) )
    
    random1 = color1
    random2 = color2
    ## Rejection sample until the palettes are different
    while lch_too_close( random2, color2 ) or lch_too_close( random2, random1 ):
        print( "rejection sampling" )
        random2 = generate_one_random_lch()
    
    return ( [ lch2rgb(c) for c in (color1,color2) ], [ lch2rgb(c) for c in (random1,random2) ] )

def generate_monochromatic():
    '''
    Returns two palettes. The first is a monochromatic palette consisting of
    two RGB colors, one with full chroma (saturation) and another with the same
    lightness and hue but half the chroma.
    The second palette will share one of the colors, but the other will have random hue.
    '''
    
    color1 = generate_one_random_lch()
    # color2 = ( color1[0], 0.5*color1[1], color1[2] )
    ## Don't go all the way to black.
    color2 = ( (0.5 + 0.25*random.random())*color1[0], (0.5 + 0.25*random.random())*color1[1], color1[2] )
    
    random1 = color1
    ## Use color1's chromaticity.
    # random2 = ( color1[0], color1[1], generate_one_random_lch()[2] )
    ## Rejection sample until the palettes are different
    random2 = color2
    while(
        lch_too_close( random2, color2 )
        or
        lch_too_close( random1, random2 )
        or
        ## Don't accidentally choose a monochromatic or complementary scheme.
        lch_too_close( random1, (random1[0], random1[1], random2[2]) )
        or
        lch_too_close( random1, (random1[0], random1[1], fmod( random2[2] + pi, 2*pi )) )
        ):
        print( "rejection sampling" )
        ## Don't go all the way to black.
        random2 = ( color2[0], color2[1], generate_one_random_lch()[2] )
    
    return ( [ lch2rgb(c) for c in (color1,color2) ], [ lch2rgb(c) for c in (random1,random2) ] )

def generate_triad():
    '''
    Returns two palettes. The first is a complementary palette consisting of
    three RGB colors equally spaced in hue around the Lch color wheel.
    The second palette will share one of the colors, but the others will have random hue.
    All colors will have identical luminance and chroma.
    '''
    
    color1 = generate_one_random_lch()
    color2 = ( color1[0], color1[1], fmod( color1[2] + 2*pi/3., 2*pi ) )
    color3 = ( color1[0], color1[1], fmod( color1[2] + 4*pi/3., 2*pi ) )
    
    ## Keep the first random color the same as the first triad color.
    random1 = color1
    ## Rejection sample until we get three distinct colors, no two of which are similar
    ## to the template palette or to each other.
    random2 = color1
    random3 = color1
    while(
            ( lch_too_close( random2, color1 ) or lch_too_close( random2, color2 ) or lch_too_close( random2, color3 ) )
            and
            ( lch_too_close( random3, color1 ) or lch_too_close( random3, color2 ) or lch_too_close( random3, color3 ) )
        ) or (
            lch_too_close( random1, random2 ) or lch_too_close( random1, random3 ) or lch_too_close( random2, random3 )
        ):
        print( "rejection sampling" )
        random2 = generate_one_random_lch()
        random3 = generate_one_random_lch()
    
    return ( [ lch2rgb(c) for c in (color1,color2,color3) ], [ lch2rgb(c) for c in sort_lch_colors_relative_to_first([random1,random2,random3]) ] )

def generate_square():
    '''
    Returns two palettes. The first is a complementary palette consisting of
    three RGB colors equally spaced in hue around the Lch color wheel.
    The second palette will share one of the colors, but the others will have random hue.
    All colors will have identical luminance and chroma.
    '''
    
    color1 = generate_one_random_lch()
    color2 = ( color1[0], color1[1], fmod( color1[2] + 1*pi/4, 2*pi ) )
    color3 = ( color1[0], color1[1], fmod( color1[2] + 2*pi/4, 2*pi ) )
    color4 = ( color1[0], color1[1], fmod( color1[2] + 3*pi/4, 2*pi ) )
    
    random1 = color1
    ## Rejection sample until we get three distinct colors, no three of which are similar
    ## to the template palette or to each other.
    random2 = color1
    random3 = color1
    random4 = color1
    while(
            ( lch_too_close( random2, color1 ) or lch_too_close( random2, color2 ) or lch_too_close( random2, color3 ) or lch_too_close( random2, color4 ) )
            and
            ( lch_too_close( random3, color1 ) or lch_too_close( random3, color2 ) or lch_too_close( random3, color3 ) or lch_too_close( random3, color4 ) )
            and
            ( lch_too_close( random4, color1 ) or lch_too_close( random4, color2 ) or lch_too_close( random4, color3 ) or lch_too_close( random4, color4 ) )
        ) or (
            lch_too_close( random1, random2 )
            or
            lch_too_close( random1, random3 )
            or
            lch_too_close( random1, random4 )
            or
            lch_too_close( random2, random3 )
            or
            lch_too_close( random2, random4 )
            or
            lch_too_close( random3, random4 )
        ):
        print( "rejection sampling" )
        random2 = generate_one_random_lch()
        random3 = generate_one_random_lch()
        random4 = generate_one_random_lch()
    
    return ( [ lch2rgb(c) for c in (color1,color2,color3,color4) ], [ lch2rgb(c) for c in sort_lch_colors_relative_to_first([random1,random2,random3,random4]) ] )

def palette2swatch( palette ):
    '''
    Given a sequence of sRGB colors, returns an image with side-by-side
    50x50 solid color squares of each color in the palette.
    '''
    
    EDGE_SIZE = 50
    
    return hstack([ asfarray(c)[None,None,:]*ones((EDGE_SIZE,EDGE_SIZE,3)) for c in palette ])

def save_palette2wedges( palette, filename, clobber = False ):
    '''
    Given a sequence of sRGB colors, saves an image with the palette
    colors in a wedge arrangement.
    '''
    
    if os.path.exists( filename ) and not clobber:
        raise RuntimeError( "File exists, will not clobber: %s" % filename )
    
    import cairocffi as cairo
    
    ### We want to arrange the colors in a circle.
    
    ### Option 1:
    ### So that colors don't overlap, let's make them circular swatches around the circle.
    ### What radius for the circle? It depends on how many colors. To keep the
    ### size constant, assume the maximum is 4. Then the closest two colors are at,
    ### without loss of generality, (r,0) and (0,r). We want the center-to-center
    ### distance to be some constant C.
    ### Then the distance sqrt(2 r^2) = sqrt(2)*r = 2*C/2 = C.
    ### So r = C/sqrt(2).
    # assert 1 <= len( palette ) <= 4
    
    ### Option 2:
    ### As wedges around a circle.
    from math import pi, cos, sin
    EDGE_SIZE = 300
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, EDGE_SIZE, EDGE_SIZE)
    ctx = cairo.Context(surface)
    
    ## Set up a transformation so that 0,0 is the center of the image,
    ## and +-1 are the corners.
    ## Subtract 10 pixels from each coordinate for padding.
    ctx.scale( EDGE_SIZE, EDGE_SIZE )
    ctx.scale( 0.5, 0.5 )
    ctx.translate( 1.0, 1.0 )
    
    arc_radians = 2.*pi/len(palette)
    for i, color in enumerate( palette ):
        ctx.set_source_rgb( *color )
        ctx.move_to( 0,0 )
        ctx.line_to( cos( (i+0)*arc_radians ), sin( (i+0)*arc_radians ) )
        ctx.arc( 0, 0, 1.0, (i+0)*arc_radians, (i+1)*arc_radians )
        ctx.line_to( cos( (i+1)*arc_radians ), sin( (i+1)*arc_radians ) )
        ctx.fill()
    
    surface.write_to_png( filename )
    print( "Saved:", filename )

def save_palette2circles( palette, filename, clobber = False ):
    '''
    Given a sequence of sRGB colors, saves an image with the palette
    colors in a circular arrangement.
    '''
    
    if os.path.exists( filename ) and not clobber:
        raise RuntimeError( "File exists, will not clobber: %s" % filename )
    
    import cairocffi as cairo
    from math import pi, cos, sin, sqrt
    
    ### We want to arrange the colors in a circle.
    
    ### Option 1:
    ### So that colors don't overlap, let's make them circular swatches around the circle.
    ### What radius for the circle? It depends on how many colors. To keep the
    ### size constant, assume the maximum is 4. Then the closest two colors are at,
    ### without loss of generality, (r,0) and (0,r). We want the center-to-center
    ### distance to be some constant C (which can be the diameter of the small circles
    ### so that they are "kissing" at those positions).
    ### Then the distance sqrt(2*r^2) = sqrt(2)*r = C.
    ### So r = C/sqrt(2).
    ### Then the image itself has width E = C + 2*r.
    ### This means that r = (E - C)/2 = C/sqrt(2).
    ### <=> E - C = C*sqrt(2) <=> C + C*sqrt(2) = E <=> C = E/(1 + sqrt(2))
    ### In our canonical coordinate system, E = 2.
    assert 1 <= len( palette ) <= 4
    EDGE_SIZE = 300
    small_circle_radius = 1./(1 + sqrt(2))
    large_circle_radius = 2*small_circle_radius/sqrt(2)
    
    ### Option 2:
    ### Find a radius for the large circle that may not take up the entire
    ### image but keeps some circles kissing (possibly with padding).
    ### TODO
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, EDGE_SIZE, EDGE_SIZE)
    ctx = cairo.Context(surface)
    
    ## Set up a transformation so that 0,0 is the center of the image,
    ## and +-1 are the corners.
    ## Subtract 10 pixels from each coordinate for padding.
    ctx.scale( EDGE_SIZE, EDGE_SIZE )
    ctx.scale( 0.5, 0.5 )
    ctx.translate( 1.0, 1.0 )
    
    radians_apart = 2*pi/len(palette)
    for i, color in enumerate( palette ):
        ctx.set_source_rgb( *color )
        cx, cy = large_circle_radius*cos(i*radians_apart), large_circle_radius*sin(i*radians_apart)
        ctx.move_to( cx, cy )
        ctx.arc( cx, cy, small_circle_radius, 0, 2*pi )
        ctx.fill()
    
    surface.write_to_png( filename )
    print( "Saved:", filename )

def generate_debug_wheel():
    '''
    Returns an sRGB image of all hues around the Lch color wheel.
    '''
    
    out_of_gamut_count = 0
    
    img = zeros((50,360,3))
    c = asfarray(generate_one_random_lch())
    for col, h in enumerate( linspace( 0, 2*pi, 360 ) ):
        c[2] = h
        img[:,col,:] = asfarray(lch2rgb(c))[None,:]
        
        ## verify
        if not lch_color_is_in_sRGB_gamut( c ):
            print( "Color out of sRGB gamut!" )
            out_of_gamut_count += 1
    
    print( "Wheel: %s colors out of gamut (%.2f%%)" % ( out_of_gamut_count, (100.*out_of_gamut_count) / 360 ) )
    
    return img

def save_image_to_file( img, filename, visualization = 'squares', clobber = False ):
    if os.path.exists( filename ) and not clobber:
        raise RuntimeError( "File exists, will not clobber: %s" % filename )
    
    skimage.io.imsave( filename, img )
    print( "Saved:", filename )

def save_palette_pair( prefix, template, clobber = False, template2filename = None ):
    if template == 'complementary':
        t, r = generate_complementary()
    elif template == 'monochromatic':
        t, r = generate_monochromatic()
    elif template == 'triad':
        t, r = generate_triad()
    elif template == 'square':
        t, r = generate_square()
    else:
        raise RuntimeError( "Unknown template: %s" % template )
    
    if template2filename is None:
        tname = prefix + '-' + template + '.png'
        rname = prefix + '-' + 'random' + '.png'
    else:
        tname = template2filename( prefix, template )
        rname = template2filename( prefix, 'random' )
    
    if visualization == 'squares':
        timg = palette2swatch( t )
        rimg = palette2swatch( r )
        save_image_to_file( timg, tname, clobber = clobber )
        save_image_to_file( rimg, rname, clobber = clobber )
    elif visualization == 'wedges':
        save_palette2wedges( t, tname, clobber = clobber )
        save_palette2wedges( r, rname, clobber = clobber )
    elif visualization == 'circles':
        save_palette2circles( t, tname, clobber = clobber )
        save_palette2circles( r, rname, clobber = clobber )
    else:
        raise RuntimeError( "Unknown visualization: %s" % visualization )
    
    return tname, rname

def main():
    import argparse
    parser = argparse.ArgumentParser( description = "Save a pair of palettes, one from a template and one random." )
    parser.add_argument( "template", type = str, choices = ['complementary', 'monochromatic', 'triad', 'square'], help="The template. Must be one of: complementary, monochromatic, triad, square." )
    parser.add_argument( "prefix", type = str, help="The output path prefix." )
    parser.add_argument( "--clobber", action = 'store_true', help="If set, this will overwrite existing files." )
    parser.add_argument( "-L", type = float, help="Override the default luminance." )
    parser.add_argument( "-c", type = float, help="Override the default chroma." )
    parser.add_argument( "--wheel", action = 'store_true', help="Generate a debug wheel saved to 'wheel.png'." )
    parser.add_argument( "--seed", type = int, help="The random seed. Default 1." )
    parser.add_argument( "-V", "--visualization", type = str, default = 'squares', help="The visualization style. Default 'squares'." )
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed( args.seed )
    
    global LUMINANCE, CHROMA
    if args.L is not None:
        LUMINANCE = args.L
    if args.c is not None:
        CHROMA = args.c
    
    print( "Luminance:", LUMINANCE )
    print( "Chroma:", CHROMA )
    
    if args.wheel:
        W = generate_debug_wheel()
        save_image_to_file( W, 'wheel.png', clobber = args.clobber )
    
    save_palette_pair( args.prefix, args.template, visualization = args.visualization, clobber = args.clobber )

if __name__ == '__main__':
    main()
