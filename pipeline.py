import numpy as np
import os
import pandas as pd
import pingouin as pg
import tempfile
import wandb
import yaml

from argparse import ArgumentParser
from pathlib import Path


__DEVOPS_BUILDNUMBER = "BUILD_BUILDNUMBER"


def get_temp_dir():
    return Path(get_env_var("__DEVOPS_TEMPDIRECTORY", tempfile.gettempdir()))


def get_project(branch):
    if branch == "":
        return "dev"
    return ".".join([w for w in branch.split("/") if w not in ["refs", "heads"]])


def get_env_var(name, default=None):
    return os.environ[name] if name in os.environ else default


def running_in_ci():
    """Returns True is the current processes is running as part of the continuous intergation pipeline"""
    return get_env_var(__DEVOPS_BUILDNUMBER) is not None


def export_variable(key, variable):
    print(f"##vso[task.setvariable variable={key}]{variable}")


def log_error(msg):
    print(f"##vso[task.logissue type=error]{msg}")


def fail_build(msg=""):
    print(f"##vso[task.complete result=Failed;]{msg}")


def get_sweep_dataframe(sweep):
    summary_list = list()
    config_list = list()
    name_list = list()
    index = list()
    for run in sweep.runs:
        index.append(run.id)
        summary_list.append(run.summary._json_dict)
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        config_list.append(config)
        name_list.append(run.name)

    summary_df = pd.DataFrame.from_records(summary_list, index=index)
    config_df = pd.DataFrame.from_records(config_list, index=index)
    name_df = pd.DataFrame({"name": name_list}, index=index)

    return pd.concat([name_df, config_df, summary_df], axis=1)


def get_data(runs):
    summary_list = list()
    config_list = list()
    name_list = list()
    index = list()
    for run in runs:
        index.append(run.id)
        summary_list.append(run.summary._json_dict)
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        config_list.append(config)

    summary = pd.DataFrame.from_records(summary_list, index=index)
    config = pd.DataFrame.from_records(config_list, index=index)

    comp_cols = filter(
        lambda c: c.split("/")[0].lower()
        in ["datamodule", "model", "downsampler", "trainer"],
        list(config.columns),
    )
    comp_cols = filter(
        lambda c: c not in {"optim/betas", "trainer/logger/name"}, comp_cols
    )
    comp_cols = filter(lambda c: len(set(config[c].dropna())) > 1, comp_cols)
    comp_cols = list(comp_cols)

    df = config[comp_cols]
    df["acc"] = summary.best_dt_acc
    df = df.sort_values(comp_cols)

    return df, comp_cols


def create_heatmap(results):
    values = list(set(results.A) | set(results.B))
    indices = {c: i for i, c in enumerate(values)}
    n = len(values)

    matrix = np.ones((n, n))
    for _, row in results.iterrows():
        p = row["p-unc"]
        a = indices[row.A]
        b = indices[row.B]

        matrix[a, b] = p
        matrix[b, a] = p

    return matrix, values


def eval_runs(entity, project, build):
    print("Staring Evaluation...")
    print(f"Entity: {entity}")
    print(f"Project: {project}")
    print(f"Build: {build}")
    api = wandb.Api()

    runs = [
        run
        for run in api.runs(f"{entity}/{project}")
        if str(build) in list(map(str, run.tags))
    ]

    df, cols = get_data(runs)

    tags = [build]
    run_name = f"eval:{project}/{build}"
    with wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=tags,
        group=build,
        job_type="evaluation",
    ) as run:
        for c in cols:
            within = pg.pairwise_ttests(
                dv="acc", within=c, data=df, padjust="sidak", return_desc=True
            )
            between = pg.pairwise_ttests(
                dv="acc", between=c, data=df, padjust="sidak", return_desc=True
            )
            tukey = pg.pairwise_tukey(dv="acc", between=c, data=df)

            run.log(
                {
                    c + "/within/ttest": wandb.Table(dataframe=pd.DataFrame(within)),
                    c + "/between/ttest": wandb.Table(dataframe=pd.DataFrame(between)),
                    c + "/between/tukey": wandb.Table(dataframe=pd.DataFrame(tukey)),
                }
            )


def parse_args(parser):
    parser.add_argument("--project", type=str)
    parser.add_argument("--build", type=str)
    parser.add_argument("--entity", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    args = parse_args(parser)
    eval_runs(args.entity, args.project, args.build)
