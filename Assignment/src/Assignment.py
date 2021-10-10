import argparse
import os

import cv2 as cv
import number_finder

def main():
    # Resolve command line args
    args = resolveArgs()

    # Use args to parse required data
    resolvePaths(args)

    if args.task == 'task1':
        number_finder.task1(
            testImgs=args.testImgs,
            outputDir=args.outputDir,
            digitsDict=args.digitsDict
        )
    else: # task2
        number_finder.task2(
            testImgs=args.testImgs,
            outputDir=args.outputDir,
            digitsDict=args.digitsDict
        )

def resolvePaths(args):
    # OS file path separator (\ or /)
    sep = os.path.sep

    # Get the current project directory path (src/../)
    projDir = f'{os.path.dirname(os.path.abspath(__file__))}{sep}..{sep}'
    
    # Directory of images to test
    if args.task == 'task1':
        testDir = f'{args.test}{sep}task1{sep}'
    else:
        testDir = f'{args.test}{sep}task2{sep}'

    # Directory for output of program
    if args.task == 'task1':
        args.outputDir = f'{args.output}{sep}task1{sep}'
    else:
        args.outputDir = f'{args.output}{sep}task2{sep}'
    # Ensure the output directory exists
    os.makedirs(args.outputDir, exist_ok=True)

    # Number of images under test directory
    numTrain = args.num
    # Images to test
    if args.task == 'task1':
        args.testImgs = [cv.imread(f'{testDir}BS{imgNum:02d}.jpg') for imgNum in range(1, numTrain + 1)]
    else:
        args.testImgs = [cv.imread(f'{testDir}DS{imgNum:02d}.jpg') for imgNum in range(1, numTrain + 1)]

    digitsDir = f'{projDir}digits{sep}'
    digitsDict = {
        0: 'Zero',
        1: 'One',
        2: 'Two',
        3: 'Three',
        4: 'Four',
        5: 'Five',
        6: 'Six',
        7: 'Seven',
        8: 'Eight',
        9: 'Nine',
        'l': 'LeftArrow',
        'r': 'RightArrow'
    }
    # Maps from a key from digitsDict to a list template images
    args.digitsDict = {
        key: [cv.imread(f'{digitsDir}{name}{i}.jpg') for i in range(1,6)] for key, name in digitsDict.items()
    }

def resolveArgs():
    parser = argparse.ArgumentParser(
        description='MP Assignment by Alec Maughan',
        add_help=True,
        epilog='dependencies: python 3.7.3, opencv-python 3.4.2.16, matplotlib'
    )

    parser.add_argument(
        'task',
        help='task1 (Building numbers) or task2 (Directional numbers)',
        type=str,
        choices=('task1', 'task2'),
    )

    parser.add_argument(
        '--test', '-t',
        help='Directory of images to process',
        type=str
    )

    parser.add_argument(
        '--output', '-o',
        help='Directory of output',
        type=str
    )

    parser.add_argument(
        '--num', '-n',
        help='Number of images under the test directory',
        type=int
    )

    return parser.parse_args()

if __name__ == '__main__':
    main()
