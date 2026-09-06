from setuptools import find_packages, setup

package_name = "embodied_agent_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Embodied Agent Developer",
    maintainer_email="developer@example.com",
    description="ROS 2 transport bridge for the embodied agent starter.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "action_bridge = embodied_agent_ros.action_bridge:main",
            "loopback = embodied_agent_ros.loopback_demo:main",
            "simulation = embodied_agent_ros.simulation_server:main",
        ],
    },
)
