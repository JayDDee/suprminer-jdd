# Suprminer-jdd

Suprminer-jdd is a fork of sp-suprminer that adds support for X16RT algorithm, fixes compile errors on newer
compilers, adds support for Ampere compute 8.6 builds, and adds some additional performance optimizations
to the entire X16 family.

It is avaiable as source code only. It can be built on Linux like most other versions of ccminer.
Building on Windows using MSVC requires a small update to the project files. If someone can provide
those changes, by pull or diff, they will be included.

Only minimal support is being provided. There are no plans for further development. There will be no formal
releases an no Windows binaries will be provided.

It has been tested on Maxwell, Pascal and Turing on Ubuntu-20.04, and Ampere on Ubuntu-21.10, using
the Ubuntu multiverse CUDA repository. Newer GPUs have no pre-defined intensity tuning, the default
intensity may not be optimal, use -i to tune manually.

--------------------------------

# ccminer

suprminer sp-mod (september 2019) optimized x16r/x16rv2/x17 algo without any dev fee.

Most optimizations come from sp, so please support him.

Overclock the core and memory for the best performance

This variant was tested and built on Linux (ubuntu server 14.04, 16.04, Fedora 22 to 25)
It is also built for Windows 7 to 10 with VStudio 2013, to stay compatible with Windows 7 and Vista.

Note that the x86 releases are generally faster than x64 ones on Windows, but that tend to change with the recent drivers.

The recommended CUDA Toolkit version is 9.2
------------------------------

This project requires some libraries to be built :

- OpenSSL (prebuilt for win)
- Curl (prebuilt for win)
- pthreads (prebuilt for win)

The tree now contains recent prebuilt openssl and curl .lib for both x86 and x64 platforms (windows).

To rebuild them, you need to clone this repository and its submodules :
    git clone https://github.com/peters/curl-for-windows.git compat/curl-for-windows


Compile on Linux
----------------

Please see [INSTALL](https://github.com/tpruvot/ccminer/blob/linux/INSTALL) file or [project Wiki](https://github.com/tpruvot/ccminer/wiki/Compatibility)
