import datajoint as dj

import pipeline.lab as lab
import pipeline.experiment as experiment
from pipeline.pipeline_tools import get_schema_name

schema = dj.schema(get_schema_name('videography'))

@schema
class BehaviorCamera(dj.Lookup):
    definition = """
    camera_position:                    varchar(20)     # side / bottom / ..
    """
    contents = zip(('side', 'bottom'))

    
@schema
class BehaviorVideo(dj.Imported):
    definition = """
    -> experiment.SessionTrial
    -> BehaviorCamera
    ---
    video_frame_count:           int             #
    video_frame_rate:           float
    video_frame_times:          longblob             # center of the exposition, relative to trial start, in seconds
    video_exposition_time:       float           # in ms
    video_folder:              varchar(200)
    vide_file_name:           varchar(100)
    """
    
@schema
class DLCTracking(dj.Imported):
     definition = """
    -> BehaviorVideo
    bodypart :varchar(50)
    ---
    x : longblob #x coordinate in pixels
    y : longblob #y coordinate in pixels
    p : longblob # likelihood
    
    """
@schema
class EmbeddingVector(dj.Imported):
    definition = """
    -> experiment.SessionTrial
    embedding_dimension : int
    ---
    embedding_vector : longblob
    """
@schema
class DLCLickBout(dj.Imported):
     definition = """
    -> experiment.SessionTrial
    lick_bout_id                  : int                  # number of bout in trial              
    ---
    lick_bout_start_time          : float                # relative to trial start
    lick_bout_start_frame         : int                  # relative to trial start
    lick_bout_end_frame           : int                  # relative to trial start
    lick_bout_peak_time           : float                # relative to trial start
    lick_bout_peak_frame          : int                  # relative to trial start
    lick_bout_amplitude           : float                # pixels  
    lick_bout_amplitude_x         : float                # pixels  
    lick_bout_amplitude_y         : float                # pixels  
    lick_bout_half_width          : float                # seconds
    lick_bout_rise_time           : float                # seconds
    """
@schema
class DLCLickBoutContact(dj.Imported):
     definition = """   
     -> DLCLickBout
     ---
     -> experiment.ActionEvent
     contact_frame_number  : int                   #the frame where the contact happened
     """


@schema
class TrialEventFrame(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> experiment.TrialEvent
    -> BehaviorCamera
    ---
    frame_num : int               #  
    dt        : double            # frame_time - event time (s) 
    """
@schema
class ActionEventFrame(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> experiment.ActionEvent
    -> BehaviorCamera
    ---
    frame_num : int               #   
    dt        : double            # frame_time - event time (s) 
    """
