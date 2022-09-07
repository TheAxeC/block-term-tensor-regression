import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

url=''

setuptools.setup(
    name='bttr',
    version='0.0.1',
    author='Axel Faes',
    author_email='axel.faes@gmail.com',
    description='Block-Term Tensor Regression',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=url,
    project_urls = {
        "Bug Tracker": url+"/issues"
    },
    license='',
    packages=['bttr'],
    install_requires=['numpy', 'scipy', 'tensorly'],
)