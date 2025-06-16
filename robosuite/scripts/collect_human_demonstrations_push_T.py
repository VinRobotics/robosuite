"""
A script to collect a batch of human demonstrations and upload them to Hugging Face Hub in LeRobot format.

"""

import argparse
import json
import os
import time
import pickle

import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import login


def collect_human_trajectory(env, device, arm, max_fr, episode_cnt, directory):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
        episode_cnt (int): episode number to save the data
        directory (str): directory to save the data to
    Returns:
        bool: True if the episode was successful, False otherwise
    """

    all_obs = env.reset()

    image = get_transparent_view(env)
    obs = all_obs["robot0_joint_pos"]
    env.render()

    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    step_data = {}
    episode_data = []

    upload_data = False

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        active_robot = env.robots[device.active_robot]

        # Get the newest action
        input_ac_dict = device.input2action()

        if input_ac_dict is not None:
            upload_data = input_ac_dict["right_gripper"][0] == 1
        else:
            upload_data = False
        if upload_data:
            return (False, upload_data)

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            break

        from copy import deepcopy

        action_dict = deepcopy(input_ac_dict)  # {}
        # set arm actions
        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):  # input type passed to joint_action_policy
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type

            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        all_next_obs, _, _, _ = env.step(env_action)
        next_image = get_transparent_view(env)
        next_obs = all_next_obs["robot0_joint_pos"]

        step_data["obs"] = obs
        step_data["action"] = env_action.flatten()
        step_data["image"] = image
        episode_data.append(step_data.copy())

        env.render()

        obs = next_obs
        image = next_image

        if env._check_success():
            file_name = f"episode_{episode_cnt}.pkl"
            file_name = os.path.join(directory, file_name)
            with open(file_name, "wb") as f:
                pickle.dump(episode_data, f)
            print(f"Episode {episode_cnt} Success, Saved")
            return (True, upload_data)

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    # cleanup for end of data collection episodes
    env.close()
    return (False, upload_data)


def get_transparent_view(env):
    """Get birdview image with the robot being transparent

    Args:
        env (MujocoEnv): environment to control
    """
    original_rgba = env.sim.model.geom_rgba.copy()

    # Make robot arm and gripper transparent
    for i, name in enumerate(env.sim.model.geom_names):
        if (
            name.startswith("robot0_g")
            or name.startswith("robot0_link")
            or name.startswith("gripper0_")
            or name.startswith("fixed_mount0_")
            or name.startswith("floor")
            or name.startswith("peg")
        ):
            env.sim.model.geom_rgba[i][3] = 0.0

    img = env.sim.render(camera_name="birdview", width=256, height=256).copy()
    env.sim.model.geom_rgba[:] = original_rgba

    return img


def upload_to_huggingface(directory, repo_name):
    """Upload the collected demonstrations to Hugging Face Hub.

    Args:
        directory (str): Directory containing the demonstration files.
        repo_name (str): Name of the Hugging Face repository.
    """
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    os.system(f"rm -rf /root/.cache/huggingface/lerobot/{repo_name}")

    print(f"Uploaded demonstrations to https://huggingface.co/datasets/{repo_name}")

    # Dataset format
    features = {}

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["j0", "j1", "j2", "j3", "j4", "j5", "j6"]},
    }

    features["action"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": {"displacement": ["a0", "a1", "a2", "a3", "a4", "a5", "a6"]},
    }

    features["observation.images.camera_top_down"] = {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for file_name in os.listdir(directory):
        directory_path = os.path.join(directory, file_name)
        with open(directory_path, "rb") as f:
            print(f"Loading episode data from {file_name}")
            episode_data = pickle.load(f)

            episode_len = len(episode_data)
            for j in range(episode_len):
                frame_data = {}
                frame_data["observation.state"] = episode_data[j]["obs"].astype(np.float32)
                frame_data["action"] = episode_data[j]["action"].astype(np.float32)
                frame_data["observation.images.camera_top_down"] = episode_data[j]["image"]
                dataset.add_frame(frame_data, task="push_t")
        dataset.save_episode()
    dataset.push_to_hub(
        private=False,
        push_videos=True,
        create_repo=True,
        license="apache-2.0",
    )


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Hugging Face repository name to upload the demonstrations to, e.g., 'hugging-face-id/push_t'",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="PushT")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples)",
    )
    parser.add_argument("--device", type=str, default="logitech_gf310")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=True,
        help="(DualSense/Logitech Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)",
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        # mink-speicific import. requires installing mink
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=True,
        render_camera=args.camera,  # use agentview camera for viz
        ignore_done=True,
        use_camera_obs=True,
        camera_names=["birdview"],  # use birdview camera for image observation
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dualsense":
        from robosuite.devices import DualSense

        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    elif args.device == "logitech_gf310":
        from robosuite.devices import LogitechGF310

        device = LogitechGF310(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    elif args.device == "mjgui":
        assert args.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    episode_cnt = 0

    # collect demonstrations
    while True:
        ret, upload_data = collect_human_trajectory(env, device, args.arm, args.max_fr, episode_cnt, new_dir)
        episode_cnt += ret
        if upload_data:
            print("Upload data to Hugging Face...")
            upload_to_huggingface(new_dir, args.repo)
            break
