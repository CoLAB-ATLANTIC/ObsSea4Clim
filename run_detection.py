"""Use temperature model to do climate projections.""" 
#nohup python3 -u run_detection.py --reset_data --latlon 10,80,90,-70 > mhwdetect.log #25,75,-40,70

import shutil
from absl import logging, app

from src.config import FLAGS
from src.detection_tru_windows import detection_tru_windows

import src.utils as utils
import src.config as config
import src.filtering_and_labelling as filter

def main(_):
    """Use temperature model to do climate projections for given dates."""
    logging.info(f"Running detection model")

    if FLAGS.reset_data: utils.reset_folders()

    logging.info(f"Get input data")
    ds, x, y = utils.get_input_data(FLAGS.input_data_folder, FLAGS.start_date,
                               FLAGS.end_date, FLAGS.latlon)
    
    logging.info(f"Quantize input data")
    ds = utils.quantize_data(ds, FLAGS.min_intensity)

    last_lbl = utils.manage_last_label(FLAGS.reset_data)

    logging.info(f"Detecting events from {FLAGS.start_date} to {FLAGS.end_date}")
    detection_tru_windows(ds, last_lbl)

    #add label mapping for labels that didnt appear in the overlap
    utils.adjust_label_mapping(last_lbl)

    # go through the windows and get the start and end date of each labelled
    #  area, so that we can cut them (separate) afterwards
    logging.info(f"Saving event timesteps")
    filter.get_timesteps_per_area(config.OUTPUT_PATH)

    logging.info(f"Splice events and creating ID")
    filter.splice_and_id_events(x, y)

    logging.info(f"Events Saved. Run Completed")


if __name__ == "__main__":
    app.run(main)
