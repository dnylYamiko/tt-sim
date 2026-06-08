from setuptools import setup

package_name = "fanuc_table_tennis_mpc"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/table_tennis_mpc.launch.py"]),
    ],
    install_requires=["setuptools", "numpy", "scipy", "matplotlib"],
    zip_safe=True,
    maintainer="taona",
    maintainer_email="danieltaona@gmail.com",
    description="FANUC CRX-25iA table-tennis robot using SINDy-enhanced position MPC",
    license="MIT",
    entry_points={
        "console_scripts": [
            "fanuc_table_tennis_mpc_node = fanuc_table_tennis_mpc.fanuc_table_tennis_node:main",
            "fake_ball_publisher = fanuc_table_tennis_mpc.fake_ball_publisher:main",
            "offline_simulation = fanuc_table_tennis_mpc.offline_simulation:main",
        ],
    },
)
