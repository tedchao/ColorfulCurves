import sys
import os

import numpy as np
import cv2

from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *

from labcolorpicker import getColor, ColorPicker, useLightTheme

from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import pyqtgraph.exporters

from skimage import io, color
from PIL import Image

from func.utils import * 

## A workaround for a bug in Qt with Big Sur. Later Qt versions don't need this workaround.
## I don't know which version.
import platform
if platform.system() == 'Darwin' and platform.mac_ver()[0] >= '10.16':
    os.environ['QT_MAC_WANTS_LAYER'] = '1'

class MainWindow( QWidget ):
    
    def __init__( self ):
        super().__init__()
        
        self.title = 'Sparse Editing'
        self.x = 300
        self.y = 0
        self.width = 200
        self.height = 100
        
        self.initUI()
    
    def initUI( self ):
        self.setWindowTitle( self.title )
        self.setGeometry( self.x, self.y, self.width, self.height )
        
        self.curvefunction = pg.PlotWidget()
        self.curvefunction.setFixedSize(500, 500)
        self.curvefunction.setBackground('w')
        self.curvefunction.setXRange(0, 1, padding=0)
        self.curvefunction.setYRange(0, 1, padding=0)
        self.makeClickable_curves( self.curvefunction )
        
        # combo box for users to select palette color to change curve
        self.select_palette_for_curve = QComboBox(self)
        
        self.color_diags = QColorDialog()
        self.color_picker = ColorPicker( lightTheme = True, hideButtons = True )
        
        self.initialize_variables()
        self.labels()
        self.text_labels()
        self.boxes()
        self.buttons()
        self.separated_lines()
        
        self.widget_setup()
        self.layout_setup()
    
    def initialize_variables( self ):
        self.imagePath = ""
        self.image = None   # save image in cv2 format
        self.L0 = None
        self.N = 100    # number of discrete points needed to evaluate bilaplacian
        
        self.palette_num = 4
        
        self.palette = None
        self.palette_img = None
        self.weights = None
        self.L_biLap = None
        
        #### Reset Variables
        self.og_image = None
        self.og_palette = None
        
        # chosen pixels
        self.changed_pixel_x = -1
        self.changed_pixel_y = -1
        self.changed_pixel_red = -1
        self.changed_pixel_green = -1
        self.changed_pixel_blue = -1
        
        self.cur_color_rgb = None
        self.cur_color_lab = None
        
        # user-picked variables
        self.palette_indx = 0
        self.curve_palette_indx = 0
        
        self.p_cons_ind = 0
        self.pixel_ind = 0
        self.lum_ind = 0
        
        # default silder variables
        self.a_slider_val = 0
        self.b_slider_val = 0
        
        # previous picked constraint
        self.picker = None
        
        self.palette_cons_indicator = False
        self.image_cons_indicator = False
        self.lum_cons_indicator = False
        
        # all constraints
        self.constraint_lums = []           # Like: [ (L1, L1'), (L2, L2'), ... ]
        self.constraint_locs = []           # Like: [ (x1, y1), (x2, y2), ... ]
        self.constraint_colors = []         # Like: [ (a1, b1), (a2, b2), ... ]
        self.constraint_weights = []        # Like: [ ( ...w1... ), ( ...w2... ), ... ]
        self.constraint_palette = []        # Like: [ [(a1, b1), j1, (x1, y1), (l1', a1', b1')], [(a2, b2), j2, (x2, y2), (l2', a2', b2')], ... ]
        self.constraint_curves = []         # Like: [ (palette 0, 0.4, 0.2), (palette 1, 0.3, 0.5), ... ]
        # Note: each palette constraint has a full (lab) in its end because of the need of correct visualization on pixel panel
        # optimization does not involve the L-channel for palette
        
    def labels( self ):
        # image label
        self.imageLabel_test = QLabel()
        self.imageLabel = QLabel()
        self.makeClickable_img( self.imageLabel )
        
        # palette label
        self.paletteLabel = QLabel()
        self.makeClickable_palette( self.paletteLabel )
        
        ## label for selected pixel
        self.pixelLabel_selected = QLabel()
        self.pixelLabel_selected.setMaximumWidth( 120 )
        self.pixelLabel_selected.setStyleSheet( "background-color: black" ) 
        
        ## label for changed pixel
        self.pixelLabel_changed = QLabel()
        self.pixelLabel_changed.setAlignment( Qt.AlignCenter )
        self.makeClickable_pixel( self.pixelLabel_changed )
        self.pixelLabel_changed.setStyleSheet( "background-color: black" ) 
    
    def text_labels( self ):
        ## selected and changed pixel color
        self.select_color_original_text = QLabel( 'Original ' )
        self.select_color_original_text.setAlignment( Qt.AlignCenter )
        self.select_color_new_text = QLabel( 'New ')
        self.select_color_new_text.setAlignment( Qt.AlignCenter )
        
        self.combotext = QLabel( 'Selected Palette Color:' )
        
    def boxes( self ):
        self.btns_io_box = QVBoxLayout() # set bottons' box for I/O
        self.btns_output_box = QVBoxLayout() # set bottons' box for saving images or palettes
        
        self.all_select_color_box = QGridLayout()   # box for all selected color boxes
        
        self.combo_curve_box = QVBoxLayout()
        self.reset_btn_box = QVBoxLayout()
    
    def buttons( self ):
        def set_button_utils( button, func, text, width ):
            button.clicked.connect( func )
            button.setToolTip( text )
            
            if len( width ) == 1:
                button.setMaximumWidth( width[0] )
            else:
                button.setMinimumWidth( width[0] )
                button.setMaximumWidth( width[1] )
        
        ## button for selecting an input image
        self.img_btn = QPushButton( 'Load Image...' )
        set_button_utils( self.img_btn, self.get_image, 'Press the button to <b>select</b> an image.', (130,) )
        
        ## button for saving edited image
        self.save_btn = QPushButton( 'Save Image' )
        set_button_utils( self.save_btn, self.save_edit_image, 'Press the button to <b>save</b> an image.', (130,) )
        
        ## button for reseting palette
        # self.reset_select_color = QPushButton( 'Reset' )
        # set_button_utils( self.reset_select_color, self.reset_specific_cons, 'Press the button to <b>reset</b> the constraint color.', (120,) )
        
        self.delete_select_color = QPushButton( 'Delete' )
        set_button_utils( self.delete_select_color, self.delete_specific_cons, 'Press the button to <b>delete</b> the selected constraint.', (120,) )
        
        ## button for saving 2D palette
        self.save_palette_btn = QPushButton( 'Save Palette' )
        set_button_utils( self.save_palette_btn, self.save_palette, 'Press the button to <b>save</b> the palette.', (130,) )

        ## button for saving the curve image
        self.save_curves_btn = QPushButton( 'Save Curves' )
        set_button_utils( self.save_curves_btn, self.save_curves, 'Press the button to <b>save</b> the curves.', (130,) )
        
        self.bake_btn = QPushButton( 'Bake' )  # might need to rename this variable
        set_button_utils( self.bake_btn, self.bake_changes, 'Press the button to <b>bake</b> your all previous edits', (130,) )
        
        self.undo_all_btn = QPushButton( 'Reset Everything' )
        set_button_utils( self.undo_all_btn, self.reset_all, 'Press the button to <b>undo</b> your all previous edits', (130,) )
        
    def separated_lines( self ):
        self.line1 = QFrame()
        self.line1.setFrameShape( QFrame.HLine )
        self.line1.setFrameShadow( QFrame.Raised )
        self.line1.setLineWidth(3)
        
        self.line2 = QFrame()
        self.line2.setFrameShape( QFrame.HLine )
        self.line2.setFrameShadow( QFrame.Raised )
        self.line2.setLineWidth(3)
    
    def widget_setup( self ):
        # left control panel
        self.btns_io_box.addWidget( self.img_btn )
        self.btns_io_box.addWidget( self.save_btn )
        self.btns_io_box.addWidget( self.save_palette_btn )
        self.btns_io_box.addWidget( self.save_curves_btn )
        
        # panel for pixel constraints
        self.all_select_color_box.addWidget( self.select_color_original_text, 0, 0 )
        self.all_select_color_box.addWidget( self.pixelLabel_selected, 1, 0 )
        # self.all_select_color_box.addWidget( self.reset_select_color, 2, 0 )
        
        self.all_select_color_box.addWidget( self.select_color_new_text, 0, 1 )
        self.all_select_color_box.addWidget( self.pixelLabel_changed, 1, 1 )
        # self.all_select_color_box.addWidget( self.delete_select_color, 2, 1 )
        self.all_select_color_box.addWidget( self.delete_select_color, 2, 0, 1, 2 )
        
        # direct manipulation on curves
        self.combo_curve_box.addWidget( self.combotext )
        self.combo_curve_box.addWidget( self.select_palette_for_curve )
        
        # reset palette
        self.reset_btn_box.addWidget( self.bake_btn )
        self.reset_btn_box.addWidget( self.undo_all_btn )
    
    def layout_setup( self ):
        # Set grid layout
        grid = QGridLayout()
        grid.setSpacing(10)
        
        ## File I/O
        self.btns_io_box_group = QGroupBox( "File I/O" )
        self.btns_io_box_group.setLayout( self.btns_io_box )
        self.save_btn.setEnabled( False )
        self.save_palette_btn.setEnabled( False )
        self.save_curves_btn.setEnabled( False )
        grid.addWidget( self.btns_io_box_group, 0, 0 )
        
        ## Selected Constraint
        self.selected_constraint_widgets = QGroupBox( "Selected Constraint" )
        self.selected_constraint_widgets.setLayout( self.all_select_color_box )
        self.selected_constraint_widgets.setEnabled( False )
        grid.addWidget( self.selected_constraint_widgets, 1, 0 )
        
        ### Sliders for recoloring
        self.curve_group = QGroupBox( "Curve Editing" )
        self.curve_group.setLayout( self.combo_curve_box )
        self.curve_group.setEnabled( False )
        grid.addWidget( self.curve_group, 2, 0 )
        
        ### Reset and Bake
        self.reset_btn_group = QGroupBox( "Bake or Reset" )
        self.reset_btn_group.setLayout( self.reset_btn_box )
        self.reset_btn_group.setEnabled( False )
        grid.addWidget( self.reset_btn_group, 3, 0 )
        
        ## Add the images
        grid.addWidget( self.imageLabel, 0, 1, 5, 1, Qt.AlignTop )
        grid.addWidget( self.paletteLabel, 0, 2, 5, 1 )
        grid.addWidget( self.curvefunction, 0, 3, 5, 1 )
        ## Row 4 can stretch.
        grid.setRowStretch( 4, 1 )
        
        self.setLayout(grid)
        self.show()
    
    def enable_after_load( self ):
        ## The first panel includes the always-enabled Load button.
        self.img_btn.setEnabled( False ) # Disable the load button until we reset our state properly.
        self.save_btn.setEnabled( True )
        self.save_palette_btn.setEnabled( True )
        self.save_curves_btn.setEnabled( True )
        ## The other panels:
        # Don't enable selection buttons until something is clicked.
        # self.selected_constraint_widgets.setEnabled( True )
        self.curve_group.setEnabled( True )
        self.reset_btn_group.setEnabled( True )

    ##########################################################################################
    ##############  reset functions  ############################
    ##########################################################################################
    
    def release_all_constraints( self ):
        self.constraint_lums = []
        self.constraint_locs = []
        self.constraint_colors = []
        self.constraint_weights = []
        self.constraint_palette = []
        self.constraint_curves = []
        
        # disable selected constraint widgets
        self.selected_constraint_widgets.setEnabled( False )
    
    def reset_specific_cons( self ):
        pass
        
    def delete_specific_cons( self ):
        ## Do nothing when there is nothing loaded
        if self.palette is None: return
        
        # if user clicked on palette constraint
        if self.palette_cons_indicator:
            self.constraint_palette.pop( self.p_cons_ind )
            print( 'You deleted a palette constraint.' )
        elif self.image_cons_indicator: 
            # if user clicked image-space constraint
            self.constraint_lums.pop( self.pixel_ind )
            self.constraint_locs.pop( self.pixel_ind )
            self.constraint_colors.pop( self.pixel_ind )
            self.constraint_weights.pop( self.pixel_ind )
            print( 'You deleted an image-space constraint.' )
        elif self.lum_cons_indicator: 
            # if user clicked luminance constraint
            self.constraint_curves.pop( self.lum_ind )
            print( 'You deleted a luminance constraint.' )
            
        # call optimizer
        self.optimizer()
        
        # disable selected constraint widgets
        self.selected_constraint_widgets.setEnabled( False )
    
    def bake_changes( self ):
        ## Do nothing when there is nothing loaded
        if self.palette is None: return
        
        reply = QMessageBox.question( self, 'Message', "Are you sure you want to bake-in all your edits?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No )
        
        if reply == QMessageBox.Yes:
            self.palette = np.copy( self.palette_opt )
            self.og_image = np.copy( self.image )
            self.og_palette = np.copy( self.palette )
            self.L0 = np.copy( self.image[:, :, 0] )
            self.L_biLap = np.tile( np.array( [np.linspace( 0, 1, self.N )] ).T, (1, self.palette.shape[0]) )
            self.reset_all()
    
    def reset_variables( self ):
        self.image = np.copy( self.og_image )
        self.palette = np.copy( self.og_palette )
        self.palette_opt = np.copy( self.palette )
        
        self.release_all_constraints()
    
    def reset_all( self ):
        ## Do nothing when there is nothing loaded
        if self.image is None: return
        
        reply = QMessageBox.question( self, 'Message', "Are you sure you want to reset all your edits?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No )
        if reply == QMessageBox.No: return
        
        # retrieve everything back
        self.reset_variables()
        # recon_lum =  100 * compute_new_luminance( self.L0/100, self.L_biLap, self.weights )
        recon_lum =  100 * compute_new_luminance_fast( self.L0/100, self.L_biLap, self.weights_less_grey )
        self.image = get_recon_img( self.palette, self.weights, recon_lum )
        
        self.palette_img = get_palette_img( self.palette, self.L0/100, self.weights )
        
        self.optimizer()
        
        # set image and palette image
        image_rgb = lab2rgb_fast( self.image )
        self.update_image_panel( image_rgb )
        self.set_image( self.imageLabel, image_rgb )
        self.set_image( self.paletteLabel, self.palette_img )
    
    ##########################################################################################
    ##############  Clickable panels  ############################
    ##########################################################################################
    
    ### for direct curve manipulation
    def manipulate_curve( self, text ):
        self.curve_palette_indx = int( text )
    
    ### adding image-space constraints
    def makeClickable_img( self, widget ):
        def SendClickSignal( widget, evnt ):
            # x, y coordinate of the image
            self.changed_pixel_x = int( evnt.position().x() )
            self.changed_pixel_y = int( evnt.position().y() )
            
            # if we click on same constraint, just display it
            self.pixel_ind = self.check_if_click_on_constraint()
            if self.pixel_ind != -1:
                lum = self.constraint_lums[ self.pixel_ind ][1]
                color_ab = self.constraint_colors[ self.pixel_ind ]
                color_lab = np.array( [lum, color_ab[0], color_ab[1]] )
                color_rgb = ( lab2rgb_fast( color_lab ) * 255. ).clip( 0, 255 ).astype( np.uint8 )
            
            # otherwise, we add new constraint
            else:
                lum = self.L0[ self.changed_pixel_y, self.changed_pixel_x ] # always use original L0 to map luminance
                lum_ = self.image[:, :, 0][ self.changed_pixel_y, self.changed_pixel_x ] # always use the updated luminance
                
                color_lab = self.image[ self.changed_pixel_y, self.changed_pixel_x ]
                color_rgb = ( lab2rgb_fast( color_lab ) * 255. ).round().clip( 0, 255 ).astype( np.uint8 )
                
                W = self.weights.reshape( ( self.image.shape[0], self.image.shape[1], self.palette.shape[0] ) )
                w_at_pixel = W[ self.changed_pixel_y, self.changed_pixel_x ]
                
                # add all constraint information
                #self.constraint_lums.append( color_lab[0] )
                self.constraint_lums.append( [lum, lum_] )
                self.constraint_locs.append( (self.changed_pixel_y, self.changed_pixel_x) )
                self.constraint_colors.append( color_lab[1:] )
                self.constraint_weights.append( w_at_pixel )
                
                print( '\n -- Constraint info: ')
                print( 'Add luminance constraints:\n', (lum, lum_) )
                
                # call optimizer
                self.optimizer()
            
            self.cur_color_rgb = color_rgb
            self.cur_color_lab = color_lab
            
            # change flag of indicating adding palette constraint or image-space constraint or luminance constraint
            self.palette_cons_indicator = False
            self.image_cons_indicator = True
            self.lum_cons_indicator = False
            # enable gui
            self.selected_constraint_widgets.setEnabled( True )
            
            # display
            print( '\n -- Click info: ')
            print( 'selected color (LAB): ', color_lab )
            print( 'selected color (RGB): ', color_rgb )
            
            # change visualization of pixel labels (both selected and changed)
            color_str = '#%02x%02x%02x' % ( color_rgb[0], color_rgb[1], color_rgb[2] )
            self.pixelLabel_selected.setStyleSheet( "background-color: " + color_str ) 
            self.pixelLabel_changed.setStyleSheet( "background-color: " + color_str ) 
            
            self.update_image_panel( lab2rgb_fast( self.image ) )
            
        widget.emit( SIGNAL( 'clicked()' ) )
        widget.mousePressEvent = lambda evnt: SendClickSignal( widget, evnt )
    
    ### adding palette constraints
    def makeClickable_palette( self, widget ):
        def SendClickSignal( widget, evnt ):
            self.changed_pixel_x = evnt.pos().x()
            self.changed_pixel_y = evnt.pos().y()
            
            # compute palette index
            xoffset = (widget.width() - widget.pixmap().width())//2
            yoffset = (widget.height() - widget.pixmap().height())//2
            pos = ( self.changed_pixel_y - yoffset, self.changed_pixel_x - xoffset )
            # there are `self.palette.shape[0]` palette colors
            swatch_height = np.shape(self.palette_img)[0] // self.palette.shape[0]
            self.palette_indx = int(pos[0]) // swatch_height
            
            # Center the constraint in the palette
            pos = int( (self.palette_indx + 0.5) * swatch_height ), int( widget.pixmap().width()/2 )
            
            ## Return if the click is outside the palette.
            if self.palette_indx < 0 or self.palette_indx >= self.palette.shape[0]: return
            
            self.p_cons_ind = self.check_which_palette_constraint()
            # if currently there is no palette constraint at clicked palette
            if  self.p_cons_ind == -1:
                print( ' You place a palette constraint at palette: ' + str( self.palette_indx ) )
                
                # add palette constraint
                cons_palette = [ self.palette[self.palette_indx], self.palette_indx, pos, self.palette[self.palette_indx] ]
                self.constraint_palette.append( cons_palette )
                
                # change visualization of pixel labels (both selected and changed)
                color_rgb = ( 255. * lab2rgb_fast( np.array([50, self.palette[self.palette_indx][0], self.palette[self.palette_indx][1]]) ) ).clip( 0, 255 ).astype( np.uint8 )
                
                color_str = '#%02x%02x%02x' % ( color_rgb[0], color_rgb[1], color_rgb[2] )
                
                self.pixelLabel_selected.setStyleSheet( "background-color: " + color_str ) 
                self.pixelLabel_changed.setStyleSheet( "background-color: " + color_str ) 
            
            # if user click on existing constraint
            else:
                print( ' You choose a palette constraint at palette: ' + str( self.palette_indx ) )
                
                # change into newly selected palette constraint
                color_rgb_prev = ( 255. * lab2rgb_fast( np.array([50, self.constraint_palette[self.p_cons_ind][0][0], self.constraint_palette[self.p_cons_ind][0][1]]) ) ).clip( 0, 255 ).astype( np.uint8 )
                color_rgb = ( 255. * lab2rgb_fast( np.array([50, self.constraint_palette[self.p_cons_ind][-1][0], self.constraint_palette[self.p_cons_ind][-1][1]]) ) ).clip( 0, 255 ).astype( np.uint8 )
                
                color_str_prev = '#%02x%02x%02x' % ( color_rgb_prev[0], color_rgb_prev[1], color_rgb_prev[2] )
                color_str_changed = '#%02x%02x%02x' % ( color_rgb[0], color_rgb[1], color_rgb[2] )
                
                self.pixelLabel_selected.setStyleSheet( "background-color: " + color_str_prev ) 
                self.pixelLabel_changed.setStyleSheet( "background-color: " + color_str_changed ) 
                
            # change current clicked color to be reflected on sparse edits
            self.cur_color_rgb = color_rgb
            self.cur_color_lab = color.rgb2lab( color_rgb )
            
            # change flag of indicating adding palette constraint or image-space constraint or luminance constraint
            self.palette_cons_indicator = True
            self.image_cons_indicator = False
            self.lum_cons_indicator = False
            # enable gui
            self.selected_constraint_widgets.setEnabled( True )
            
            # call optimizer
            self.optimizer()
            
        widget.emit( SIGNAL( 'clicked()' ) )
        widget.mousePressEvent = lambda evnt: SendClickSignal( widget, evnt )
    
    # allow users to do direct curve manipulation
    def makeClickable_curves( self, widget ):
        def ProcessEventAndReturnLumXY( evnt ):
            self.changed_pixel_x = evnt.pos().x()
            self.changed_pixel_y = evnt.pos().y()
            
            lum_x = ( self.changed_pixel_x - 25 ) / 475
            lum_y = ( 475 - self.changed_pixel_y ) / 475
            
            return ( lum_x, lum_y )
        
        def SendClickSignal( evnt ):
            ## Do nothing when there is nothing loaded
            if self.image is None: return
            
            ## If we clicked, then we're dragging until mouse release.
            widget.mouseMoveEvent = SendDragSignal
            
            lum_x, lum_y = ProcessEventAndReturnLumXY( evnt )
            
            self.lum_ind = self.check_if_click_on_lum( lum_x, lum_y )
            if self.lum_ind == -1:      # if users click on current constraint, renew it
                self.lum_ind = len( self.constraint_curves )
                self.constraint_curves.append( (self.curve_palette_indx, lum_x, lum_y) )
            
            # change flag of indicating adding palette constraint or image-space constraint or luminance constraint
            self.palette_cons_indicator = False
            self.image_cons_indicator = False
            self.lum_cons_indicator = True
            # enable gui
            self.selected_constraint_widgets.setEnabled( True )
            
            # call optimizer
            self.optimizer()
        
        def SendDragSignal( evnt ):
            print( "In drag" )
            if self.image is None: return
            
            lum_x, lum_y = ProcessEventAndReturnLumXY( evnt )
            
            # Update the constraint.
            self.constraint_curves[ self.lum_ind ] = (self.curve_palette_indx, lum_x, lum_y)
            
            # call optimizer
            self.optimizer()
        
        ## UPDATE: We may be able to simply use mouseMoveEvent if
        ##         we call widget.setMouseTracking( False ). I thought it was the default,
        ##         but it doesn't seem to be.
        def SendMoveSignal( evnt ): pass
        def SendReleaseSignal( evnt ): widget.mouseMoveEvent = SendMoveSignal
        
        widget.emit( SIGNAL( 'clicked()' ) )
        # Mouse press sets up mouse move.
        widget.mousePressEvent = SendClickSignal
        
        # Mouse release resets mouse move.
        widget.mouseReleaseEvent = SendReleaseSignal
        
    def makeClickable_pixel( self, widget ):
        def SendClickSignal( widget, evnt ):
            self.select_color()
            
        widget.emit( SIGNAL( 'clicked()' ) )
        widget.mousePressEvent = lambda evnt: SendClickSignal( widget, evnt )
    
    ##########################################################################################
    ##############  Sparse Edits  ############################
    ##########################################################################################
    
    # Sparse Editing
    def select_color( self ):
        ## Do nothing when there is no selected color
        if self.cur_color_lab is None: return
        
        self.color_picker.currentColorChanged.connect( self.sparse_edit )
        self.color_picker.getColor( (self.cur_color_lab[0], self.cur_color_lab[1], self.cur_color_lab[2]) )
        '''
        self.color_diags.open()
        self.color_diags.setCurrentColor( QColor( self.cur_color_rgb[0], self.cur_color_rgb[1], self.cur_color_rgb[2] ) )
        self.color_diags.blockSignals( True )
        self.color_diags.currentColorChanged.connect( self.sparse_edit )
        self.color_diags.blockSignals( False )
        '''
        
    
    # Function for sparse edit
    def sparse_edit( self, pick ):
        pick = np.array( pick ).astype(float)
        c = (lab2rgb_fast( pick.reshape( 1, 1, -1 ) ).squeeze() * 255.).astype(int)
        
        #self.pixelLabel_changed.setStyleSheet( "background-color: " + '#%02x%02x%02x' % ( pick.red(), pick.green(), pick.blue() ) )
        self.pixelLabel_changed.setStyleSheet( "background-color: " + '#%02x%02x%02x' % ( c[0], c[1], c[2] ) )
        
        np.set_printoptions(suppress=True)
        #pick = color.rgb2lab( np.array( [ pick.red(), pick.green(), pick.blue() ] ) / 255. )
        
        if not (self.picker == pick).all():
            print( '------------------------------------------' )
            print( 'Luminance constraints:\n', np.array(self.constraint_lums)/100 )
            print( 'Weights at constraints:\n', self.constraint_weights )
            print( 'Palette constraints:\n', self.constraint_palette )
            print( 'Curve constraints:\n', self.constraint_curves )
            print( 'Color constraints:\n', self.constraint_colors )
            
            # change constraint information based on the type of constraints user clicks
            if self.image_cons_indicator:
                # separate colors and luminance
                self.constraint_colors[ self.pixel_ind ] = pick[1:]
                self.constraint_lums[ self.pixel_ind ][1] = pick[0]
                
            if self.palette_cons_indicator:
                self.constraint_palette[ self.p_cons_ind ][-1] = pick[1:]
            
            # call optimizer
            self.optimizer()
            
            # reset picker to avoid duplicate optimization
            self.picker = pick
    
    def optimizer( self ):
        print( "\n\n==== Running optimizer ====" )
        ## Who triggered this? Get rid of duplicate calls.
        # import traceback
        # traceback.print_stack()
        
        # optimization on all kinds of constraints
        self.palette_opt, self.L_biLap = optimization_pixel_constraints( self.palette, self.constraint_palette, self.constraint_curves, self.constraint_weights, self.constraint_colors, self.constraint_lums )
        
        # compute reconstructed luminance
        # recon_lum = 100 * compute_new_luminance( self.L0/100, self.L_biLap, self.weights ).clip(0,1)
        recon_lum = 100 * compute_new_luminance_fast( self.L0/100, self.L_biLap, self.weights_less_grey ).clip(0,1)
        self.image = get_recon_img( self.palette_opt, self.weights, recon_lum )
        palette_img = get_palette_img( self.palette_opt, self.L0/100, self.weights )
        
        # draw updated data onto panel
        image_rgb = lab2rgb_fast( self.image )
        self.set_image( self.imageLabel, image_rgb )
        self.set_image( self.paletteLabel, palette_img )
        
        self.update_image_panel( image_rgb )
        self.update_palette_panel( palette_img )
        
        # update curve function plot accordingly
        self.plot_curves()

    def plot_curves( self ):
        self.curvefunction.clear()
        L = np.linspace( 0, 1, self.L_biLap.shape[0] )
        
        for i in range( self.L_biLap.shape[1] ):
            # change curve color according to palette color with 50 luminance
            marker = np.array( [[ 50., self.palette_opt[i][0], self.palette_opt[i][1] ]] )
            marker = (lab2rgb_fast( marker ) * 255.).clip( 0, 255 ).astype( np.uint8 ).reshape(-1)
            
            pen = pg.mkPen( color=marker, width=5 )
            self.curvefunction.showGrid( x = True, y = True )
            self.curvefunction.plot( L, self.L_biLap[:, i], pen=pen )
            
            # plot curve constraints
            for cc in self.constraint_curves:
                if cc[0] == i:
                    self.curvefunction.plot( np.array( [cc[1]] ), np.array( [cc[2]] ), pen=pen, symbol='o' )
            
    ##########################################################################################
    ############## I/O functions ############################
    ##########################################################################################
    def convert_to_cart_coord( self, image ):
        # Given: image in LHS-space
        # Return: image_ in LXY-space
        image_ = np.zeros( image.shape )
        image_[:, :, 0] = image[:, :, 0]
        image_[:, :, 1] = np.multiply( np.cos(image[:, :, 1]), image[:, :, 2] )
        image_[:, :, 2] = np.multiply( np.sin(image[:, :, 1]), image[:, :, 2] )
        return image_
    
    # Function for loading an input image
    def get_image( self ):
        img = QFileDialog.getOpenFileName( self, 'Select file' )
        if len( img[0] ) > 0:
            path = img[0]
            print( "Loading Image:", path )
            
            # load image...
            self.image = color.rgb2lab( np.array( Image.open( path ).convert('RGB') ) / 255. )         # In LAB-space #
            
            '''
            # Crop to square if the image is wider than tall.
            # UPDATE: No need anymore.
            if self.image.shape[1] > self.image.shape[0]:
                print( "Image wider than tall. Cropping to square." )
                print( "Old dimensions:", self.image.shape )
                self.image = self.image[:,:self.image.shape[0]]
                print( "New dimensions:", self.image.shape )
            '''
            
            # If the image is too large to fit on screen, shrink it.
            MAX_WIDTH = 500
            if self.image.shape[1] > MAX_WIDTH:
                print( f"Image wider than {MAX_WIDTH} pixels. Scaling down." )
                print( "Old dimensions:", self.image.shape )
                scale = MAX_WIDTH/self.image.shape[1]
                self.image = skimage.transform.resize( self.image, ( scale * self.image.shape[0], MAX_WIDTH ) )
                print( "New dimensions:", self.image.shape )
                
                print( "Save cropped version." )
                self.imagePath = path
                self.save_image( 1 )
                
            og = self.image
            self.imagePath = path
            self.L0 = self.image[:, :, 0]
            
            # extract palette and weights at the time inputting image
            self.palette = extract_AB_palette( self.image, self.palette_num )
            self.weights = extract_weights( self.palette, self.image )
            
            #self.palette = np.zeros( (5,2) )    # grayscale approach
            self.weights_less_grey = compute_new_luminance_less_grey_weights( self.weights )
            
            # replace original image with reconstructed one
            self.L_biLap = np.tile( np.array( [np.linspace( 0, 1, self.N )] ).T, (1, self.palette.shape[0]) )
            # recon_lum =  100 * compute_new_luminance( self.L0/100, self.L_biLap, self.weights )
            recon_lum =  100 * compute_new_luminance_fast( self.L0/100, self.L_biLap, self.weights_less_grey )
            self.image = get_recon_img( self.palette, self.weights, recon_lum )       # In LAB-space #
            
            # display error
            print( 'Reconstruction error: ', np.linalg.norm( og-self.image ) / (self.image.shape[0] * self.image.shape[1]) )
            self.palette_img = get_palette_img( self.palette, self.L0/100, self.weights )
            
            # copy for reseting
            self.og_image = np.copy( self.image )
            self.og_palette = np.copy( self.palette )
            self.palette_opt = np.copy( self.palette )
            self.og_L_biLap = np.copy( self.L_biLap )
            
            # set reconstructed image and palette image
            self.set_image( self.imageLabel, lab2rgb_fast( self.image ) )
            self.set_image( self.paletteLabel, self.palette_img )
            
            # add items to combox
            for i in range( self.palette.shape[0] ):
                self.select_palette_for_curve.addItem( 'Color '+ str(i+1) )
            self.select_palette_for_curve.activated.connect( self.manipulate_curve ) 
            
            # enable controls
            self.enable_after_load()
            
            # call optimizer
            self.optimizer()
    
    # Set image on the panels
    def set_image( self, panel, image ):
        #Load the image into the label
        height, width, dim = image.shape
        image = np.asarray( ( image*255. ).round().clip( 0, 255 ).astype( np.uint8 ) )
        qim = QImage( image.data, width, height, 3 * width, QImage.Format_RGB888 )
        panel.setPixmap( QPixmap( qim ) )
        panel.repaint()
    
    # function to save current image
    def save_edit_image( self ):
        ## Do nothing when there is nothing loaded
        if self.image is None: return
        
        self.save_image( 1 )
        
    # functions to save current image
    def save_palette( self ):
        ## Do nothing when there is nothing loaded
        if self.image is None: return
        
        self.save_image( 2 )

    # function to save current curves plot
    def save_curves( self ):
        ## Do nothing when there is nothing loaded
        if self.image is None: return
        
        self.save_image( 3 )
    
    def save_image( self, option ):
        """
        option controls what is saved. 1 saves the image, 2 saves the palette, and 3 saves snapshots of the curves
        """
        if option == 1:
            s = 'image'
            saved_img = ( lab2rgb_fast( self.image ) * 255. ).round().clip( 0, 255 ).astype( np.uint8 ) 
        elif option == 2:
            s = 'palette'
            saved_img = self.palette_img
        else:
            s = 'curves'
            # https://pyqtgraph.readthedocs.io/en/latest/user_guide/exporting.html
            plot_item = self.curvefunction.getPlotItem()
            exporter = pg.exporters.ImageExporter(plot_item)
        
        
        if self.imagePath == '':
            QMessageBox.warning( self, 'Warning', 'Please select an image first.' )
        else:
            image_name = QFileDialog.getSaveFileName( self, 'Save ' + s )
            if len( image_name[0] ) == 0:
                print( "User cancelled save." )
                return
            
            if image_name[0][-4:].lower() in ['.jpg', '.png']:
                path_name = image_name[0]
            else:
                path_name = image_name[0] + '.png'

            # handle plot export separately
            if option == 3 and exporter:
                exporter.export(path_name)
                return
                
            if saved_img.dtype != np.uint8:
                # then the image is floating-point
                # but PIL Image requires 8-bit integer
                saved_img *= 255
                saved_img = saved_img.astype(np.uint8)

            Image.fromarray( saved_img ).save( path_name )
    
    ##########################################################################################
    ############## Panels update ############################
    ##########################################################################################
    
    def update_image_panel( self, image ):
        height, width, dim = image.shape
        image = np.asarray( ( image*255. ).round().clip( 0, 255 ).astype( np.uint8 ) )
        qim = QImage( image.data, width, height, 3 * width, QImage.Format_RGB888 )
        
        pixmap = QPixmap( qim )
        qp = QPainter( pixmap )
        
        for loc in self.constraint_locs:
            x, y = loc[1], loc[0]
            
            # black outer
            pen = QPen( Qt.black, 2 )
            qp.setPen( pen )
            qp.drawEllipse(x-2, y-2, 10, 10)
        
            # white filler
            pen = QPen( Qt.white, 2 )
            qp.setPen( pen )
            qp.drawEllipse(x, y, 6, 6)
        
            # black interior
            pen = QPen( Qt.black, 2 )
            qp.setPen( pen )
            qp.drawEllipse(x+2,y+2, 2, 2)
                
        qp.end()
        self.imageLabel.setPixmap( pixmap )
        self.imageLabel.repaint()
    
    def update_palette_panel( self, palette ):
        height, width, dim = palette.shape
        palette = np.asarray( ( palette*255. ).round().clip( 0, 255 ).astype( np.uint8 ) )
        qim = QImage( palette.data, width, height, 3 * width, QImage.Format_RGB888 )
        
        pixmap = QPixmap( qim )
        qp = QPainter( pixmap )
        
        for pc in self.constraint_palette:
            x, y = pc[2][1], pc[2][0]
            
            # black outer
            pen = QPen( Qt.black, 2 )
            qp.setPen( pen )
            qp.drawEllipse(x-2, y-2, 10, 10)
            
            # white filler
            pen = QPen( Qt.white, 2 )
            qp.setPen( pen )
            qp.drawEllipse(x, y, 6, 6)
            
            # black interior
            pen = QPen( Qt.black, 2 )
            qp.setPen( pen )
            qp.drawEllipse(x+2,y+2, 2, 2)
            
        qp.end()
        self.paletteLabel.setPixmap( pixmap )
        self.paletteLabel.repaint()
        
    ##########################################################################################
    ############## Process functions ############################
    ##########################################################################################
    
    # check if user clicks on duplicate constraint
    def check_if_click_on_constraint( self ):
        click = ( self.changed_pixel_y, self.changed_pixel_x )
        
        dist = 10
        for i in range( len(self.constraint_locs) ):
            # if user clicks samewhere around the existing constraint, then we return
            loc = self.constraint_locs[i]
            if np.linalg.norm( np.array(click) - np.array(loc) ) <= dist:
                return i
        return -1
    
    def check_which_palette_constraint( self ):
        for i in range( len(self.constraint_palette) ):
            pc = self.constraint_palette[i]
            if pc[1] == self.palette_indx:
                return i
        return -1
    
    def check_if_click_on_lum( self, lum_x, lum_y ):
        dist = 0.05
        for i in range( len(self.constraint_curves) ):
            # if user clicks on the same curve around same constraint
            if self.constraint_curves[i][0] == self.curve_palette_indx:
                if np.abs( lum_x - self.constraint_curves[i][1] ) < dist:
                    return i
        return -1
    
    # Function if users tend to close the app
    def closeEvent( self, event ):
        reply = QMessageBox.question( self, 'Message', "Are you sure you want to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication( sys.argv )
    ex = MainWindow()
    sys.exit( app.exec_() )
    
    
if __name__ == '__main__':
    main()
