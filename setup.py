from setuptools import setup, find_packages

    
setup(name='savo',
    version=0.0,
    description='Surrogate gradient Assisted Very-high-dimensional Optimization',
    classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Programming Language :: Python :: 3.6',
    'Topic :: FRIB beam tuning'
    ],
    keywords = ['high-dim','high dimensional','machine-learning', 'optimization', 'FRIB beam tuning'],
    author='Kilean Hwang',
    author_email='hwang@frib.msu.edu',
#     license='',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'botorch',
        'machineIO',
        #'git+https://github.com/kilean20/machineIO.git',
    ],
    zip_safe=False)
