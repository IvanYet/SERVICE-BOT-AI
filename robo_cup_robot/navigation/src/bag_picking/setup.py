from glob import glob
from setuptools import setup

package_name = 'bag_picking'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alson_wong',
    maintainer_email='alsonwong280@gmail.com',
    description='Paper bag detection and picking',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'bag_locator = bag_picking.bag_locator:main',
            'bag_picker = bag_picking.bag_picker:main',
            'ir_capture = bag_picking.ir_capture:main',
        ],
    },
)
