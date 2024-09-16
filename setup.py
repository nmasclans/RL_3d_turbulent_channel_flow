from setuptools import setup, find_packages

with open('README.md') as f:
    readme_file = f.read()

with open('LICENSE.txt') as f:
    license_file = f.read()

setup(
    name='smartrhea',
    version='1.0',
    description='A ML framework for the RHEA CFD solver using SmartSim',
    long_description=readme_file,
    author='Nuria Masclans',
    author_email='nuria.masclans@upc.edu',
    keywords=['reinforcement learning, computational fluid dynamics'],
    url='',
    license=license_file,
    packages=find_packages(exclude=('ProjectRHEA')),
		setup_requires=['numpy==1.21.0'],
		install_requires=['numpy==1.21.0', 'smartsim==0.4.2', 'smartredis==0.5.3', 'tensorflow==2.8.0', 'tensorflow-probability==0.16.0', 'tf_agents==0.12.0', 'matplotlib']
)
