import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DiSTNet2D",
    version="0.1.8",
    author="Jean Ollion",
    author_email="jean.ollion@polytechnique.org",
    description="tensorflow/keras implementation of DiSTNet 2D",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeanollion/distnet2d",
    download_url='https://github.com/jeanollion/distnet2d/releases/download/v0.1.8/distnet2d-0.1.8.tar.gz',
    packages=setuptools.find_packages(),
    keywords=['Segmentation', 'Tracking', 'Cell', 'Tensorflow', 'Keras'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3',
    install_requires=['numpy', 'scipy', 'tensorflow>=2.7.1', 'edt>=2.0.2', 'scikit-fmm', 'numba', 'dataset_iterator>=0.4.5', 'elasticdeform>=0.4.7']
)
