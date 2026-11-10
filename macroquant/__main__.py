import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from macroquant.experiment import run,config_load,ROOT

def main():
    parser=argparse.ArgumentParser(description="MacroQuant reproducible market-risk research")
    sub=parser.add_subparsers(dest="command",required=True)
    r=sub.add_parser("run");r.add_argument("--config");r.add_argument("--cache",default="data/cache");r.add_argument("--output",default=None)
    r=sub.add_parser("report");r.add_argument("output");r.add_argument("--config")
    r=sub.add_parser("paper");r.add_argument("--selection",default="reference_results/selection.json");r.add_argument("--cache",default="data/cache");r.add_argument("--ledger",default="paper/forecasts.sqlite");r.add_argument("--config")
    args=parser.parse_args();config=config_load(args.config)
    if args.command=="run":
        output=args.output or "results/"+datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        if (Path(output)/"summary.json").exists():raise SystemExit("This run is complete. Use a NEW output directory rather than overwrite its evidence.")
        print(json.dumps(run(config,args.cache,output),indent=2))
    elif args.command=="report":
        from macroquant.report import report
        report(config,Path(args.output));print(str(Path(args.output)/"report.html"))
    else:
        from macroquant.paper import paper
        print(json.dumps(paper(config,args.selection,args.cache,args.ledger),indent=2))
if __name__=="__main__":main()
