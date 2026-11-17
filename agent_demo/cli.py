import argparse
import uvicorn

def run(args):
    uvicorn.run(
        "agent_demo.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )

def main():
    parser = argparse.ArgumentParser(prog="ai4eosc-agent", description="Run the A2A agent demo server.",)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run the A2A server.")
    run_parser.add_argument("--host", default="0.0.0.0")
    run_parser.add_argument("--port", type=int, default=8000)
    run_parser.add_argument("--log-level", default="info")
    run_parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()