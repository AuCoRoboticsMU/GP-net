import collections
from datetime import datetime
import uuid
import argparse

import numpy as np
import pandas as pd
import tqdm
import sys
sys.path.append('.')

from src.experiments.simulation import ClutterRemovalSim
from src.experiments.detection import GPnet
from src.experiments.vis_utils import read_grasp, append_csv, create_csv
from pathlib import Path


MAX_CONSECUTIVE_FAILURES = 2
DEBUG = True
VIS = False
if VIS:
    import src.experiments.vis as vis

State = collections.namedtuple("State", ["depth_im", "camera_intrinsics", "camera_extrinsics", "points"])


def run(grasp_plan_fn, logdir, description, scene, object_set, num_objects=1, n=1,
        num_rounds=1, seed=1, sim_gui=False):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed)
    logger = Logger(logdir, description)

    for _ in tqdm.tqdm(range(num_rounds)):
        sim.reset(num_objects)
        sim.save_state()

        round_id = logger.last_round_id() + 1
        logger.log_round(round_id, sim.num_objects)

        consecutive_failures = 0
        #
        # while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES:
        timings = {}

        # scan the scene
        depth_ims, extrinsics, pc, timings["integration"] = sim.acquire_depth(n=n)

        # plan grasps
        for depth_im, extrinsic in zip(depth_ims, extrinsics):
            state = State(depth_im, sim.camera, extrinsic, pc.points)
            grasps, scores, timings["planning"] = grasp_plan_fn(state)

            if len(grasps) == 0:
                break  # no detections found, abort this round

            # execute grasp
            for grasp, score in zip(grasps, scores):
                label, _ = sim.execute_grasp(grasp, allow_contact=True, remove=False)
                sim.restore_state()

                # log the grasp
                logger.log_grasp(round_id, state, timings, grasp, score, label)

class Logger(object):
    def __init__(self, root, description):
        time_stamp = datetime.now().strftime("%H-%M-%S")
        description = "{}_{}".format(time_stamp, description).strip("_")

        self.logdir = root / description
        self.scenes_dir = self.logdir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.grasps_csv_path = self.logdir / "grasps.csv"
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "integration_time",
                "planning_time",
            ]
            create_csv(self.grasps_csv_path, columns)

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        append_csv(self.rounds_csv_path, round_id, object_count)

    def log_grasp(self, round_id, state, timings, grasp, score, label):
        # log scene
        depth_im, camera_intr, camera_extr, points = state.depth_im, state.camera_intrinsics, state.camera_extrinsics, state.points
        scene_id = uuid.uuid4().hex
        scene_path = self.scenes_dir / (scene_id + ".npz")
        np.savez_compressed(scene_path, depth_im=depth_im, intrinsics=camera_intr, extrinsics=camera_extr, points=points)

        # log grasp
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label = int(label)
        append_csv(
            self.grasps_csv_path,
            round_id,
            scene_id,
            qx,
            qy,
            qz,
            qw,
            x,
            y,
            z,
            width,
            score,
            label,
            timings["integration"],
            timings["planning"],
        )


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def num_per_object(self):
        df = (
            self.grasps[["round_id"]]
            .groupby("round_id")
            .size()
        )
        values = list(df.values)
        for i in range(len(values), 100):
            values.append(0)
        return values

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):
        scene_id, grasp, label = read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))
        print(scene_data)

        return scene_data["points"], grasp, score, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--logdir", type=Path, default="data/experiments")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--detection-threshold", type=float, default=0.1)
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="packed")
    parser.add_argument("--object-set", type=str, default="packed/test")
    parser.add_argument("--num-objects", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--model-name", type=str, default='gpnet_125.pt')
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--sim-gui", action="store_true")
    args = parser.parse_args()

    # Load model
    if args.model is None:
        test_model = 'GP-net'
    else:
        test_model = args.model
    path_dir = 'data/runs/{}/{}'.format(test_model, args.model_name)
    centre_rep = 'centre' in test_model
    grasp_planner = GPnet(Path(path_dir), centre_representation=centre_rep,
                          detection_threshold=args.detection_threshold, debug=DEBUG)

    if args.seed == -1:
        seed = np.random.randint(0, 100)
        print("Seed: ", seed)
    else:
        seed = args.seed

    if VIS:
        import rospy

        rospy.init_node("sim_grasp", anonymous=True)

    run(grasp_plan_fn=grasp_planner,
        logdir=args.logdir,
        description=args.description,
        scene=args.scene,
        object_set=args.object_set,
        num_objects=args.num_objects,
        num_rounds=args.num_rounds,
        seed=seed,
        sim_gui=args.sim_gui,
        )
