#!/usr/bin/env python3
"""Bring up the simulation, publish random targets, and make the expert look at them.

Starts:
  * shutter_lookat/start_sim.launch.py -- MuJoCo simulation plus RViz, with the
    lookat-target config so that the blue targets are visible
  * shutter_servo/shutter_servo.launch.py -- MoveIt Servo, which moves the arm
    towards the joint positions that the expert commands
  * generate_random_target.py -- a new random target every 1/publish_rate seconds
  * expert_opt.py -- the expert, which makes the robot look at each target and can
    save the state-action pairs to disk

MuJoCo owns the clock, so every node started here runs with use_sim_time:=true.

Usage:
    ros2 launch shutter_behavior_cloning collect_data.launch.py
    ros2 launch shutter_behavior_cloning collect_data.launch.py save_state_actions:=True
    ros2 launch shutter_behavior_cloning collect_data.launch.py save_state_actions:=True output_file:=<path>
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch arguments
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='0.25',
        description='Publish rate for the target generator'
    )

    save_state_actions_arg = DeclareLaunchArgument(
        'save_state_actions',
        default_value='False',
        description='Save state actions to file'
    )

    output_file_arg = DeclareLaunchArgument(
        'output_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('shutter_behavior_cloning'),
            'data',
            'state_action.txt'
        ]),
        description='Output file for state actions'
    )

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Suppress the MuJoCo Simulate window.'
    )

    sim_speed_factor_arg = DeclareLaunchArgument(
        'sim_speed_factor',
        default_value='1.0',
        description='1.0 is real time; higher runs the physics faster than real time.'
    )

    face_arg = DeclareLaunchArgument(
        'face',
        default_value='true',
        description="Render Shutter's face onto the head screen."
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Start RViz alongside the simulation.'
    )

    # MuJoCo simulation plus RViz. The lookat-target config shows the targets on /target_marker.
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('shutter_lookat'),
                'launch',
                'start_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
            'sim_speed_factor': LaunchConfiguration('sim_speed_factor'),
            'face': LaunchConfiguration('face'),
            'rviz': LaunchConfiguration('rviz'),
            'rviz_config': PathJoinSubstitution([
                FindPackageShare('shutter_lookat'),
                'config',
                'lookat-target.rviz'
            ]),
        }.items()
    )

    # MoveIt Servo moves the arm towards the joint positions commanded by the expert.
    # It is started once the simulation has finished spawning its controllers (see
    # wait_for_controllers.py); otherwise its switch to the joint_group_controller can fail.
    wait_for_controllers_node = Node(
        package='shutter_behavior_cloning',
        executable='wait_for_controllers.py',
        name='wait_for_controllers',
        output='screen'
    )

    servo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('shutter_servo'),
                'launch',
                'shutter_servo.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': 'true',
        }.items()
    )

    start_servo = RegisterEventHandler(OnProcessExit(
        target_action=wait_for_controllers_node,
        on_exit=lambda event, context: [servo_launch] if event.returncode == 0 else
        [LogInfo(msg="MoveIt Servo was not started because the simulation's controllers were not ready")]
    ))

    # Generate random target node
    generate_random_target_node = Node(
        package='shutter_behavior_cloning',
        executable='generate_random_target.py',
        name='generate_random_target',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'publish_rate': LaunchConfiguration('publish_rate')
        }]
    )

    # Expert optimization node
    expert_opt_node = Node(
        package='shutter_behavior_cloning',
        executable='expert_opt.py',
        name='expert_opt',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'output_file': LaunchConfiguration('output_file'),
            'save_state_actions': LaunchConfiguration('save_state_actions')
        }]
    )

    return LaunchDescription([
        publish_rate_arg,
        save_state_actions_arg,
        output_file_arg,
        headless_arg,
        sim_speed_factor_arg,
        face_arg,
        rviz_arg,
        sim_launch,
        wait_for_controllers_node,
        start_servo,
        generate_random_target_node,
        expert_opt_node
    ])
