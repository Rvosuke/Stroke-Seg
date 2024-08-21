from setuptools import setup, find_packages

setup(
    name='Stroke-Seg',
    version='0.1.0',
    description='A brief description of your project',
    author='Rvosuke',
    author_email='zeyangbai.rvo@gmail.com',
    url='https://github.com/Rvosuke/Stroke-Seg',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        # List your project dependencies here
        # For example:
        # 'numpy>=1.13.3',
        # 'pandas>=0.20.3',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
