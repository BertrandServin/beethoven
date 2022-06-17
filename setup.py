from setuptools import setup


def readme():
    with open("README.rst") as f:
        return f.read()


setup(
    name="beethoven",
    version="0.1",
    description="Working with bees in harmony",
    long_description=readme(),
    url="https://github.com/BertrandServin/beethoven",
    author="Bertrand Servin",
    author_email="bertrand.servin@inrae.fr",
    license="LGPLv3",
    packages=["beethoven"],
    install_requires=["numpy", "scipy", "pandas", "fastphase"],
    scripts=["bin/qg_pool","bin/genoqueen_hom"],
    zip_safe=False,
)
