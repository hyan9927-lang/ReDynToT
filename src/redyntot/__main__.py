"""Explicit stage commands; invoking help never starts inference."""
import argparse
from . import io, runtime, metrics


def main():
    parser = argparse.ArgumentParser(description="ReDynToT research stages")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-prompts")
    build.add_argument("--questions", required=True)
    build.add_argument("--config", required=True)
    build.add_argument("--output", required=True)
    generate = commands.add_parser("decompose")
    generate.add_argument("--prompts", required=True)
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", required=True)
    flatten = commands.add_parser("prepare-trees")
    flatten.add_argument("--input", required=True)
    flatten.add_argument("--profile", choices=sorted(runtime.PROFILES), required=True)
    flatten.add_argument("--output", required=True)
    answer = commands.add_parser("answer")
    answer.add_argument("--trees", required=True)
    answer.add_argument("--config", required=True)
    answer.add_argument("--corpus", required=True)
    answer.add_argument("--index-dir", required=True)
    answer.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--questions", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "build-prompts":
        runtime.build_prompts(args.questions, args.config, args.output)
    elif args.command == "decompose":
        runtime.decompose(args.prompts, args.config, args.output)
    elif args.command == "prepare-trees":
        runtime.prepare_trees(args.input, args.profile, args.output)
    elif args.command == "answer":
        runtime.answer_trees(args.trees, args.config, args.corpus, args.output, args.index_dir)
    else:
        result = metrics.evaluate(io.questions(args.questions), io.load_rows(args.predictions))
        io.save_json(args.output, result)
        print({key: value for key, value in result.items() if key != "per_question"})


if __name__ == "__main__":
    main()
