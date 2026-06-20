from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'panda_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'tf_transformations', 'opencv-python', 'numpy'],
    zip_safe=True,
    maintainer='utk',
    maintainer_email='sulaimanfawwazak@gmail.com',
    description='Color detection package for Panda robot vision (RGB object detection)',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'color_detector = panda_vision.color_detector:main',
            'color_detector_rgbd = panda_vision.color_detector_rgbd:main',
            'yolo_detector = panda_vision.yolo_detector:main',
            # 'speech_controller = panda_vision.speech_controller:main',
            'speech_listener = panda_vision.speech_listener:main',
            'llm_controller = panda_vision.llm_controller:main',
            'instrument_orchestrator = panda_vision.instrument_orchestrator:main',
        ],
    },
)