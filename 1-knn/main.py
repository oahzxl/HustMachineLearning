import argparse
import logging

from runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False,
                        help='train + test')
    parser.add_argument('--evaluate', action='store_true', default=True,
                        help='train + eval')
    parser.add_argument('--k', type=int, default=3,
                        help='k nearest neighbours')
    parser.add_argument('--path', type=str, default='data/',
                        help='data path')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='show an image\'s k nearest neighbours')
    parser.add_argument('--acc', action='store_true', default=False,
                        help='show loss of different k')
    parser.add_argument('--scale', type=float, default=0.7,
                        help='train data / total data')
    return parser.parse_args()


def main(args):
    print(args)
    runner = Runner(args)
    if args.plot:
        runner.plot()
    if args.train and not args.acc:
        logging.info("Start predicting test data...")
        runner.predict(train=True)
    if args.evaluate and not args.acc:
        logging.info("Start predicting eval data...")
        runner.predict(train=False)
    if args.acc:
        logging.info("Start predicting...")
        runner.acc()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main(parse_args())
