from setuptools import setup, find_packages
from zagros import __version__

with open("README.rst") as tmp:
    readme = tmp.read()

setup(
    author='Iniyan Natarajan',
    author_email='iniyan.natarajan@wits.ac.za',
    name='zagros',
    version=__version__,
    description='Zagros is an Application for Gauging Radio Observation Statistics',
    long_description=readme,
    long_description_content_type="text/x-rst",
    url='https://github.com/saiyanprince/zagros',
    license='GNU GPL v2',
    packages=find_packages(include=['zagros','zagros.*']),
    entry_points={
        'console_scripts': ['zagros=zagros:main']
    },
    keywords='zagros',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3.9',
        ],
)
