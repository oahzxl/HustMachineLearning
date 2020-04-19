import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--evaluate', action='store_true', default=True)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--path', type=str, default='data/')
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--scale', type=float, default=0.6)
    return parser.parse_args()


def main(args):
    print(args)
    from runner import Runner
    runner = Runner(args)
    if args.train:
        runner.predict(train=True)
    if args.evaluate:
        runner.predict(train=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main(parse_args())
