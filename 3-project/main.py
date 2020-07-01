import argparse
import logging

from runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=True,
                        help='train + test + eval')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='load model + eval')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='')
    parser.add_argument('--d-model', type=int, default=300,
                        help='')
    parser.add_argument('--max-words', type=int, default=80,
                        help='')
    parser.add_argument('--max-num-epochs', type=int, default=10,
                        help='')
    parser.add_argument('--model-saved-path', type=str, default=r"./result",
                        help='')
    return parser.parse_args()


def main(args):
    print(args)
    runner = Runner(args)
    if args.train:
        logging.info("Start training...")
        runner.train()
    if args.eval:
        logging.info("Start evaluating...")
        runner.eval()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main(parse_args())
