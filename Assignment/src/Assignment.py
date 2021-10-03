import argparse
import number_finder

def main():
    parser = argparse.ArgumentParser(
        description='MP Assignment by Alec Maughan',
        add_help=True,
    )

    parser.add_argument(
        'task',
        help='task1 (Building numbers) or task2 (Directional numbers)',
        type=str,
        choices=('task1', 'task2'),
    )

    args = parser.parse_args()

    if args.task == 'task1':
        number_finder.task1()

if __name__ == '__main__':
    main()
