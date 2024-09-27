from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cgodme",
    version="0.1.4",
    description="Find optimal path flows that map systematic variables (origin-destination demand and link flows) to ensure consistency",
    author="Taehooie Kim, Ph.D., Xin Wu, Ph.D., Han Zheng, Ph.D., Xuesong Zhou, Ph.D.",
    author_email="taehooie.kim@gmail.com, xin.wu@villanova.edu, hzheng73@asu.edu, xzhou74@asu.edu",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Taehooie/CGODME",
    include_package_data=True,
    package_data={
        'cgodme': ["config.yaml"]
    }
)