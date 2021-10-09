import argparse
import os

import cv2 as cv
import number_finder

def main():
    args = resolveArgs()

    sep = os.path.sep

    # Get the current project directory path
    projDir = f'{os.path.dirname(os.path.abspath(__file__))}{sep}..{sep}'
    
    trainDir = args.train
    task1TrainDir = f'{trainDir}task1{sep}'
    task2TrainDir = f'{trainDir}task2{sep}'

    outputDir = args.output
    task1OutputDir = f'{outputDir}task1{sep}'
    task2OutputDir = f'{outputDir}task2{sep}'

    numTrain = args.num
    task1TestImgs = [lambda: cv.imread(f'{trainDir}BS{imgNum:02d}.jpg') for imgNum in range(1, numTrain + 1)]
    task2TestImgs = [lambda: cv.imread(f'{trainDir}DS{imgNum:02d}.jpg') for imgNum in range(1, numTrain + 1)]

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

    # Maps from a key from digitsDict to a list of lambdas to read that image
    digitsDict = {
        key: [lambda: cv.imread(f'{digitsDir}{name}{i}.jpg') for i in range(1,6)] for key, name in digitsDict.items()
    }

    if args.task == 'task1':
        number_finder.task1(
            testImgs=task1TestImgs,
            outputDir=outputDir,
            digitsDict=digitsDict
        )
    else: # task2
        pass

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
