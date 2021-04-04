from pipeline.ingest import datapipeline_metadata, datapipeline_experiment
from utils import utils_pipeline

utils_pipeline.extract_files_from_dir() # exports bpod sessions
datapipeline_metadata.populatemetadata(updatemetadata = True ) # update google metadata and upload lab schema
datapipeline_experiment.populate_behavior() # populate experiment schema