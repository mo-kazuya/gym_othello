from setuptools import setup, find_packages

setup(
    name='gym_othello',
    version='0.1.0',
    description='A simple othello',
    author='Kazuya Morikubo',
    author_email='mo.kazuya@gmail.com',
    url='https://github.com/mo-kazuya/gym_othello.git',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
)