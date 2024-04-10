from config import paths

from argparse import ArgumentParser
import argparse

from os import listdir
from os.path import basename

from stream.stream import Stream

from depth.depth import Depth
from depth.depth_postprocessing import DepthPostProcessor

from path.path_planning import PathPlanner

def main():

    parser = get_parser()
    select_mode(parser=parser)

def default_message():
    
    print("\r No option specified. \n\r Type '--help' for list of arguments.")

def open_textfile(path: str) -> str:
    
    with open(path, 'r', encoding='utf-8') as file:
        description = file.read().strip()
        return description

def get_descriptions() -> list:

    return [
        open_textfile(path=paths.description_path),
        open_textfile(path=paths.calibration_description_path),
        open_textfile(path=paths.stream_description_path),
        open_textfile(path=paths.stereo_description_path)
        ]

def get_parser():
    
    (
        description_path,
        calibration_description_path,
        stream_description_path,
        stereo_description_path
    ) = get_descriptions() 

    parser = ArgumentParser(description=description_path)
    calibration_parser = parser.add_argument_group('Calibration Mode')
    calibration_parser.add_argument('-c','--calibration', action='store_true', help=calibration_description_path)

    stream_parser = parser.add_argument_group('Stream Mode')
    stream_parser.add_argument('-s', '--stream', action='store_true', help=stream_description_path)
    stream_parser.add_argument('--ssd', nargs='?', const=None, default=argparse.SUPPRESS, type=int, help="Subflag ssd with optional integer. Defaults to 1000 if not provided, or uses the provided value.")
    stream_parser.add_argument('--ss', nargs='?', const=1000, default=argparse.SUPPRESS, type=int, help="Subflag ss with optional integer. Defaults to 1000 if not provided, or uses the provided value.")
    stream_parser.add_argument('--sd', action='store_true', help="Subflag sd, a boolean flag")
    stream_parser.add_argument('--sv', nargs='?', const=1000, default=argparse.SUPPRESS, type=int, help="Subflag sv with optional integer. Defaults to 1000 if not provided, or uses the provided value.")

    stereo_parser = parser.add_argument_group('Stereo Mode')
    stereo_parser.add_argument('-d', '--stereo', action='store_true', help=stereo_description_path)
    stereo_parser.add_argument('-dp', '--stereo_postprocessing', action='store_true', help=stereo_description_path)
    stereo_parser.add_argument('-og', '--occupancy_grid', action='store_true', help=stereo_description_path)

    path_parser = parser.add_argument_group('Path Mode')
    path_parser.add_argument('-p', '--path_planning', action='store_true', help=stereo_description_path)

    return parser

def select_mode(parser):
   
    # TODO 
    args = parser.parse_args()

    if args.calibration: 
        print('calibration mode')
        
    elif args.stream:
        stream = Stream()

        if hasattr(args, 'ssd'): 
            stream.run(mode='save_display_mode', frame_limit=args.ssd)
        elif hasattr(args, 'ss'):
            stream.run(mode='save_mode', frame_limit=args.ss)
        elif hasattr(args, 'sv'):
            stream.run(mode='void_mode', frame_limit=args.sv)
        elif args.sd:
            stream.run(mode='display_mode')
        else:
            stream.run()

    elif args.stereo:
        
        calib_dir = 'calibration_parameters'
        depth_calculator = Depth(calib_dir)
        depth_calculator.display_3d_depth_map()

    elif args.stereo_postprocessing:

        calib_dir = 'calibration_parameters'
        depth_post_processor = DepthPostProcessor(calib_dir)
        depth_post_processor.display_filtered_3d_depth_map()

    elif args.occupancy_grid:

        calib_dir = 'calibration_parameters'
        depth_post_processor = DepthPostProcessor(calib_dir)
        depth_post_processor.display_xz_projection()

    elif args.path_planning:

        calib_dir = 'calibration_parameters'
        path_planner = PathPlanner(calib_dir)

    else: 
        default_message()

if __name__ == '__main__': main()

