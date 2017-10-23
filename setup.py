from setuptools import setup, find_packages

setup(
    name='jupyter-dataprocess',
    version='0.0.1',
    description='Jupyter Widgets for data cleaning and preprocessing',
    url='https://github.com/saili109/jupyter-dataprocess',
    author='Sai Li',
    author_email='sailiinuk@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'ipywidgets',
        'seaborn',
        'scikit-learn'
    ]
)

