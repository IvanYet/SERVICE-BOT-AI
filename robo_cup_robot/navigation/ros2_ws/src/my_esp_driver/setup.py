import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'my_esp_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # --- Tell ROS 2 to copy your launch, maps, and config folders on build ---
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')), # <--- ADDED!
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='oscar',
    maintainer_email='oscarsoon26@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'esp_bridge = my_esp_driver.esp_bridge:main'
        ],
    },
)