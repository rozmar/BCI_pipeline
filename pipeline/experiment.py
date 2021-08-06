import datajoint as dj

import pipeline.lab as lab
from pipeline.pipeline_tools import get_schema_name

schema = dj.schema(get_schema_name('experiment'))


@schema
class BrainLocation(dj.Manual):
    definition = """
    brain_location_name: varchar(32)  # unique name of this brain location (could be hash of the non-primary attr)
    ---
    -> lab.BrainArea
    -> lab.Hemisphere
    -> lab.SkullReference
    """

@schema
class Session(dj.Manual):
    definition = """
    -> lab.Subject
    session: smallint 		# session number
    ---
    session_date: date
    session_time: time
    unique index (subject_id, session_date, session_time)
    -> lab.Person
    -> lab.Rig
    """

@schema
class Task(dj.Lookup):
    definition = """
    # Type of tasks
    task            : varchar(50)                  # task type
    ----
    task_description : varchar(4000)
    """
    contents = [
         ('BCI OL', 'Open loop, lickport moves towards the mouse without feedback.'),
         ('BCI CL', 'Closed loop, lickport moves towards the mouse on ROI activity'),
         ('BCI CL photostim', 'Closed loop, lickport moves towards the mouse on ROI activity, photostim added.'),
         ]


@schema
class TaskProtocol(dj.Lookup):
    definition = """
    # SessionType
    -> Task
    task_protocol : tinyint # task protocol
    ---
    task_protocol_description : varchar(4000)
    """
    contents = [
         ('BCI OL', 0, 'lickport moves constantly towards the mouse until it licks'),
         ('BCI OL', 1, 'lickport moves in a random manner towards the mouse until it licks'),
         ('BCI CL', 10, 'lickport moves on ROI activity'),
         ('BCI CL', 11, 'lickport moves on ROI activity, but there is a baseline movement'),
         ('BCI CL', 12, 'lickport moves on the difference of the activity of multiple ROIs, also there is a baseline movement'),
         ('BCI CL photostim', 20, 'lickport moves on ROI activity, the conditioned neuron is photostimulated'),         
         ]

@schema
class LickPort(dj.Lookup):
    definition = """
    lick_port: varchar(16)  # e.g. left, right, middle, top-left, purple
    """
    contents = zip(['left', 'right', 'middle'])


@schema
class SessionTrial(dj.Imported):
    definition = """
    -> Session
    trial : smallint 		# trial number
    ---
    trial_start_time : decimal(10, 5)  # (s) relative to session beginning 
    trial_end_time : decimal(10, 5)  # (s) relative to session beginning 
    """

@schema 
class TrialNoteType(dj.Lookup):
    definition = """
    trial_note_type : varchar(20)
    """
    contents = zip(('autolearn', 'protocol #', 'bad', 'bitcode','autowater'))


@schema
class TrialNote(dj.Imported):
    definition = """
    -> SessionTrial
    -> TrialNoteType
    ---
    trial_note  : varchar(255) 
    """


@schema
class SessionTask(dj.Manual):
    definition = """
    -> Session
    -> TaskProtocol
    """

@schema
class SessionComment(dj.Manual):
    definition = """
    -> Session
    session_comment : varchar(767)
    """

@schema
class SessionDetails(dj.Manual):
    definition = """
    -> Session
    session_weight : decimal(8,4) # weight of the mouse at the beginning of the session
    session_water_earned : decimal(8,4) # water earned by the mouse during the session
    session_water_extra : decimal(8,4) # extra water provided after the session
    """

@schema
class TrialInstruction(dj.Lookup):
    definition = """
    # Instruction to mouse 
    trial_instruction  : varchar(8) 
    """
    contents = zip(('left', 'right', 'middle', 'none'))


@schema
class Outcome(dj.Lookup):
    definition = """
    outcome : varchar(32)
    """
    contents = zip(('hit', 'miss', 'ignore'))


@schema
class BehaviorTrial(dj.Imported):
    definition = """
    -> SessionTrial
    ----
    -> TaskProtocol
    -> TrialInstruction
    -> Outcome
    auto_water=0: bool  # water given in the beginning of the trial to facilitate licks
    free_water=0: bool  # "empty" trial with water given (go-cue not played, no trial structure) 
    """

@schema
class TrialLickportChoice(dj.Imported):
    definition = """  
    -> BehaviorTrial
    ----
    -> LickPort
    """

@schema
class TrialEventType(dj.Lookup):
    definition = """
    trial_event_type  : varchar(20)  
    """
    contents = zip(('go', 'threshold cross', 'trial end', 'lickport step','reward'))


@schema
class TrialEvent(dj.Imported):
    definition = """
    -> BehaviorTrial 
    trial_event_id: smallint
    ---
    -> TrialEventType
    trial_event_time : decimal(9, 5)   # (s) from trial start, not session start
    trial_event_duration : decimal(9,5)  #  (s)  
    """


@schema
class ActionEventType(dj.Lookup):
    definition = """
    action_event_type : varchar(32)
    ----
    action_event_description : varchar(1000)
    """
    contents =[  
       ('left lick', ''), 
       ('right lick', ''),
       ('middle lick', ''),
       ]


@schema
class ActionEvent(dj.Imported):
    definition = """
    -> BehaviorTrial
    action_event_id: smallint
    ---
    -> ActionEventType
    action_event_time : decimal(9,5)  # (s) from trial start
    """
@schema
class BCISettings(dj.Imported):
    definition = """  
    -> BehaviorTrial
    ----
    bci_conditioned_neuron_ids   :   blob
    bci_minimum_voltage_out      :   double
    bci_movement_punishment_t    :   double
    bci_movement_punishment_pix  :   double #
    """ 
    class ConditionedNeuron(dj.Part):
        definition = """
        -> master
        bci_conditioned_neuron_idx    :    smallint
        ---
        bci_conditioned_neuron_sign  :   double
        bci_threshold_low            :   double 
        bci_threshold_high           :   double
        """
@schema
class LickPortSetting(dj.Imported):
    definition = """  
    -> BehaviorTrial
    ----
    lickport_limit_far               :   decimal(5, 3) #mm #negative
    lickport_limit_close             :   decimal(5, 3) #mm #positive
    lickport_threshold               :   decimal(5, 3) #mm #zero
    lickport_step_size               :   decimal(5, 3) #mm
    lickport_step_time               :   double #s
    lickport_auto_step_freq          :   double #Hz
    lickport_maximum_speed           :   double # mm/s
    """

    class OpenDuration(dj.Part):
        definition = """
        -> master
        -> LickPort
        ---
        open_duration: decimal(5, 4)  # (s) duration of port open time
        """