import datajoint as dj
import pipeline.lab as lab#, ccf
import pipeline.experiment as experiment
from pipeline.pipeline_tools import get_schema_name
schema = dj.schema(get_schema_name('imaging'),locals())

#%%

@schema
class FOV(dj.Imported):
    definition = """
    -> experiment.Session
    fov_number                : smallint                # either multi-plane imaging or separate FOVs in a session
    ---
    fov_x_size                  : double                # (pixels)
    fov_y_size                  : double                # (pixels)
    fov_frame_rate              : double                # (Hz) - there might be gaps, median frequency           
    fov_frame_num               : int                   # number of frames in all the movies in this FOV      
    fov_pixel_size              : decimal(5,2)          # in microns
    """
@schema
class FOVMetaData(dj.Imported):
    definition = """
    -> FOV
    ---
    fov_directory               : varchar (500) #location of the raw data
    fov_movie_names             : longblob      #name of separate movies included 
    fov_movie_frame_nums        : longblob      #frame number of each movie
    fov_zoomfactor              : double
    fov_beam_power              : double
    fov_flyback_time            : double        #seconds  
    fov_lineperiod              : double        #seconds  
    fov_scanframeperiod         : double        #seconds  
    """

@schema
class FOVChannel(dj.Imported):
    definition = """
    -> FOV
    channel_number              : smallint
    ---
    channel_color               : varchar (20) #green/red
    channel_offset = NULL       : int # general offset for all movies
    channel_subtract_offset     : tinyint # 1 if offset subtracted, 0 if not
    channel_description         : varchar (200) # GCaMP/Alexa/..
    """
    
@schema
class FOVMeanImage(dj.Imported):
    definition = """
    -> FOVChannel
    ---
    fov_mean_image              : longblob                # 
    fov_mean_image_enhanced     : longblob                # 
    fov_max_projection=NULL     : longblob                #
    """


@schema
class FOVFrameTimes(dj.Imported):
    definition = """
    -> FOV
    ---
    frame_times                : longblob              # timing of each frame relative to Session start - frame start time
    """
@schema
class MotionCorrection(dj.Imported): 
    definition = """
    -> FOV
    motion_correction_id    : smallint             # id of motion correction in case of multiple motion corrections
    ---
    motion_corr_description     : varchar(300)         #description of the motion correction - rigid/nonrigid
    motion_corr_x_block=null    : longblob            
    motion_corr_y_block=null    : longblob
    motion_corr_x_offset        : longblob         # registration vectors   
    motion_corr_y_offset        : longblob         # registration vectors   
    """

@schema
class ROI(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> FOV
    roi_number                      : int           # roi number (restarts for every FOV, same number in different channels means the same ROI)
    ---
    roi_centroid_x                  : double        # ROI centroid  x, pixels
    roi_centroid_y                  : double        # ROI centroid  y, pixels
    roi_xpix                        : longblob      # pixel mask 
    roi_ypix                        : longblob      # pixel mask 
    roi_pixel_num                   : int
    roi_weights                     : longblob      # weight of each pixel
    roi_aspect_ratio                : double
    roi_compact                     : double
    roi_radius                      : double
    roi_skew                        : double
    roi_s2p_idx                     : int  
    """
    
@schema
class ROINeuropil(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> ROI
    neuropil_number                      : int  # in case there are different neuropils
    ---
    neuropil_xpix                        : longblob      # pixel mask 
    neuropil_ypix                        : longblob      # pixel mask 
    neuropil_pixel_num                   : int
    """
    
@schema
class ROITrace(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> ROI
    -> FOVChannel
    ---
    roi_f_raw                       : longblob
    roi_f_corr                      : longblob # neuropil corrected
    roi_dff                         : longblob # neuropil corrected with running F0
    roi_f_mean                      : double   # neuropil corrected
    roi_f_median                    : double   # neuropil corrected
    roi_f_min                       : double   # neuropil corrected
    roi_f_max                       : double   # neuropil corrected
    """    


@schema
class ROINeuropilTrace(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> ROINeuropil
    -> FOVChannel
    ---
    neuropil_f                           : longblob
    neuropil_f_mean                      : double
    neuropil_f_median                    : double
    neuropil_f_min                       : double
    neuropil_f_max                       : double
    """

@schema
class ConditionedROI(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> ROI
    ---
    cond_roi_index_scanimage             : int
    cond_roi_multiplier                  : double
    multi_neuron_conditioning            : int    
    """
    

@schema
class TrialEventFrame(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> experiment.TrialEvent
    -> FOV
    ---
    frame_num : int               #  
    dt        : double            # frame_time - event time (s) 
    """
@schema
class ActionEventFrame(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> experiment.ActionEvent
    -> FOV
    ---
    frame_num : int               #   
    dt        : double            # frame_time - event time (s) 
    """
    
@schema
class TrialStartFrame(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> experiment.SessionTrial
    -> FOV
    ---
    frame_num : int               #   
    dt        : double            # frame_time - event time (s) 
    """
@schema
class TrialEndFrame(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> experiment.SessionTrial
    -> FOV
    ---
    frame_num : int               #   
    dt        : double            # frame_time - event time (s) 
    """   