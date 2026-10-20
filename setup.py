from setuptools import find_packages, setup

setup(
    name="vehicle-traffic-monitor",
    version="1.0.0",
    description="YOLOv3 road traffic counting and sustained-congestion SMS alerts",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt", encoding="utf-8")
        if line.strip() and not line.startswith("#")
    ],
)
