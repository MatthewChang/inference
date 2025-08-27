from setuptools import setup, find_packages

setup(
    name='inference',
    version='0.1.0',
    packages=find_packages(include=['inference', 'inference.*']),  # Explicitly include the inference package and submodules
    install_requires=['submitit','hydra-core'],  # Add dependencies here
    include_package_data=True,
    description='A Python package for running inference parallelizable with slurm.',
    author='Matthew Chang',
    author_email='matthewchang',
    url='https://github.com/MatthewChang/inference',  # Replace with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
