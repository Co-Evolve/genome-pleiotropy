import argparse
import sys

from pleiotropy.evolution.runner import evolve, evaluate, visualize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Genome Pleiotropy")
    parser.add_argument("--mode", type=str, default="evolve",
                        help="evolve: run evolution, evaluate: evaluate a given set of genomes, "
                             "visualize: evaluate a given set of genomes with visuals.")
    parser.add_argument("--cluster", type=bool, default=False, help="Whether or not to run in cluster mode.")
    parser.add_argument("--params", type=str, default="pleiotropy/evolution/meta/params/runN.json",
                        help='Path to a parameter json file in pleiotropy/evolution/meta/params')
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--name", type=str, required=True,
                        help='Name to give the experiment (used for saving and logging).')
    parser.add_argument("--eval-path", type=str, default=None,
                        help="If mode is 'evaluate' or 'visualize', this argument specifies the path to saved genomes.")

    args = parser.parse_args()

    sys.setrecursionlimit(3200)

    if args.mode == "evolve":
        evolve(args)
    elif args.mode == "evaluate":
        assert args.eval_path is not None, "Eval mode requires the eval-path CLI argument"
        evaluate(args)
    elif args.mode == "visualize":
        assert args.eval_path is not None, "Visualization mode requires eval-path CLI argument"
        visualize(args)
    else:
        raise NotImplementedError
