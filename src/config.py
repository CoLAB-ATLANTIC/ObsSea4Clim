"""Global variables and relative paths used throught the repository."""
from datetime import timedelta
from absl import flags

# VARIABLES
window_days=200
WINDOW_SIZE = timedelta(days=window_days) #728)
STEP_SIZE = timedelta(window_days - int(window_days/4))
NEIGHBOURHOOD= [[1,1,1],
                [1,1,1],
                [1,1,1]]
CONNECTIVITY = 6

DOWNSAMPLE_RATIO = 1
RENDER_VIDEOS = True

# RELATIVE PATHS:
OUTPUT_PATH = './data/output/'
INTERNAL_DATA_PATH = './data/internal/'
VIDEO_FOLDER = OUTPUT_PATH + 'debug_videos/'
MAPPING_FOLDER = '/home/eoserver/beatriz/JP/data/'
MAPPING_FILE = INTERNAL_DATA_PATH + 'label_mapping_total.pkl'

# FLAGS
FLAGS = flags.FLAGS

flags.DEFINE_string("input_data_folder", 
                    '/media/data_HDD/joaop/MHW_detection/data/',
                    "path to folder with input data",# required=True,
                    )

flags.DEFINE_string("start_date", '2017-01-01',
                    "start date of desired interval")#, required=True)

flags.DEFINE_string("end_date", '2019-01-01',
                    "end date of desired interval")#, required=True)

flags.DEFINE_float("min_intensity", 0.2, "minimum pixel intensity for detection")

flags.DEFINE_integer("min_days", 5, "minimum consecutive days for detection")

flags.DEFINE_integer("min_pixels_frame", int(10_000/(DOWNSAMPLE_RATIO**2)),
                      "minimum number of total 2d connected pixels for detection") #10_000

flags.DEFINE_integer("min_pixels_time", int(250_000/(DOWNSAMPLE_RATIO**2)),
                      "minimum number of total 3d connected pixels for detection") #250_000

flags.DEFINE_bool("reset_data", False,
                      "reset detection process or not. so if you run again, data was maintained")

flags.DEFINE_list('latlon', None, #pt: 36, 44, -10, 1 | eu: 25,75,-40,70
                   'Latitude/Longitude range as four values: lat_min lat_max lon_min lon_max')