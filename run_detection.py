"""Use temperature model to do climate projections.""" 
#nohup python3 -u run_detection.py --reset_data --latlon 25,75,-40,70 > mhwdetect.log

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
    ds = utils.get_input_data(FLAGS.input_data_folder, FLAGS.start_date,
                               FLAGS.end_date, FLAGS.latlon)
    
    logging.info(f"Quantize input data")
    ds = utils.quantize_data(ds, FLAGS.min_intensity)

    last_lbl = utils.manage_last_label(FLAGS.reset_data)

    logging.info(f"Detecting events")
    detection_tru_windows(ds, last_lbl)

    #add label mapping for labels that didnt appear in the overlap
    utils.adjust_label_mapping(last_lbl)

    #GUARDAR ESTES OUTPUTS EM DIFERENTES PASTAS???? 
    # TIPO ./OUTPUT/LABELLED_WINDOWS E ./OUTPUT/DETECTED_AREAS... ALGO ASSIM?

    # go through the windows and get the start and end date of each labelled
    #  area, so that we can cut them (separate) afterwards
    logging.info(f"Saving event timesteps")
    filter.get_timesteps_per_area(config.OUTPUT_PATH)

    #################################################
    ####         FALTA O LABELLING E GUARDAR AS MHWS FINAIS !!!
    ####    sep_id_functions.py e separate_and_id.ipynb !!!!
    #################################################

    logging.info(f"Splice events and creating ID")
    filter.splice_and_id_events()

    print('Test done')


if __name__ == "__main__":
    app.run(main)
