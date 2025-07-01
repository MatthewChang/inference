from setuptools import setup, find_packages

setup(
    name='inference',
    version='0.1.0',
    packages=find_packages(include=['inference', 'inference.*']),  # Explicitly include the inference package and submodules
    install_requires=[],  # Add dependencies here
    include_package_data=True,
    description='A Python package for running inference using cycle diffusion processors and handling Synthia datasets.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/inference',  # Replace with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
