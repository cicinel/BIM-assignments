#!/usr/bin/env python3
"""Bring up the simulation with a moving target, filter its observations, and make the robot follow it.

Starts:
  * shutter_lookat/start_sim.launch.py -- MuJoCo simulation plus RViz, with this
    package's follow-target config (target observations and Kalman filter markers)
  * shutter_servo/shutter_servo.launch.py -- MoveIt Servo, which moves the arm
    towards the joint positions that the expert commands
  * generate_continuous_target.py -- observations of a moving target on /target
  * kalman_filter.py -- your Kalman filter, which publishes /filtered_target
  * expert_opt.py -- the expert from shutter_behavior_cloning, which makes the robot
    look at the target on follow_target_topic

MuJoCo owns the clock, so every node started here runs with use_sim_time:=true.
Keep sim_speed_factor at 1.0: MoveIt Servo runs on the wall clock, so it cannot keep
up with a target that moves faster than real time.

Usage:
    ros2 launch shutter_kf follow_target.launch.py
    ros2 launch shutter_kf follow_target.launch.py add_noise:=true path_type:=circular
    ros2 launch shutter_kf follow_target.launch.py follow_target_topic:=/true_target
    ros2 launch shutter_kf follow_target.launch.py add_noise:=true path_type:=circular observation_delay_frames:=12
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
        default_value='30',
        description='Publish rate for the target generator'
    )

    path_type_arg = DeclareLaunchArgument(
        'path_type',
        default_value='horizontal',
        description='Path type for target movement (horizontal, vertical or circular)'
    )

    add_noise_arg = DeclareLaunchArgument(
        'add_noise',
        default_value='false',
        description='Add noise to target observations'
    )

    fast_target_arg = DeclareLaunchArgument(
        'fast_target',
        default_value='false',
        description='Make target move faster'
    )

    observation_delay_frames_arg = DeclareLaunchArgument(
        'observation_delay_frames',
        default_value='0',
        description='Number of frames by which the target observations on /target are delayed (/true_target is not delayed)'
    )

    follow_target_topic_arg = DeclareLaunchArgument(
        'follow_target_topic',
        default_value='/filtered_target',
        description='PoseStamped topic that the robot follows (e.g., /filtered_target or /true_target)'
    )

    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Suppress the MuJoCo Simulate window.'
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

    # MuJoCo simulation plus RViz
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
            'face': LaunchConfiguration('face'),
            'rviz': LaunchConfiguration('rviz'),
            'rviz_config': PathJoinSubstitution([
                FindPackageShare('shutter_kf'),
                'config',
                'follow-target.rviz'
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

    # Generate continuous target node
    generate_continuous_target_node = Node(
        package='shutter_kf',
        executable='generate_continuous_target.py',
        name='generate_continuous_target',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'publish_rate': LaunchConfiguration('publish_rate'),
            'path_type': LaunchConfiguration('path_type'),
            'add_noise': LaunchConfiguration('add_noise'),
            'fast': LaunchConfiguration('fast_target'),
            'observation_delay_frames': LaunchConfiguration('observation_delay_frames')
        }]
    )

    # Kalman filter node
    kalman_filter_node = Node(
        package='shutter_kf',
        executable='kalman_filter.py',
        name='kalman_filter',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Expert optimization node
    expert_opt_node = Node(
        package='shutter_behavior_cloning',
        executable='expert_opt.py',
        name='expert_opt',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'save_state_actions': False
        }],
        remappings=[
            ('/target', LaunchConfiguration('follow_target_topic'))
        ]
    )

    return LaunchDescription([
        publish_rate_arg,
        path_type_arg,
        add_noise_arg,
        fast_target_arg,
        observation_delay_frames_arg,
        follow_target_topic_arg,
        headless_arg,
        face_arg,
        rviz_arg,
        sim_launch,
        wait_for_controllers_node,
        start_servo,
        generate_continuous_target_node,
        kalman_filter_node,
        expert_opt_node
    ])
