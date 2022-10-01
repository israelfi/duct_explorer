# Gallery Explorer
This is a ROS package for autonomous navigation and topological mapping for a robot inspecting confined enviroments.

This package is divided into three main nodes, as described in the following sections.

## gallery_explorer.py
This is the node responsible for the robot control. 
It is a vector field based navigation system whose main purpose is to make the robot follow a tangent path alongside a wall.

The considered cinematic model of the robot is the unycle, i.e.:

$$\begin{equation}
\begin{aligned}
&\dot{x}=v \cos (\theta), \\
&\dot{y}=v \sin (\theta), \\
&\dot{\theta}=\omega,
\end{aligned}
\label{eq:robotModel}
\end{equation}$$
