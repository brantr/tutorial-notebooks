# tutorial-notebooks
Tutorial Jupyter Notebooks for Data Preview 0, created and maintained by the Rubin Observatory Community Engagement Team.

<br>

| Title  | Description  |
|---|---|
| 01. Intro to DP0 Notebooks | Use a python notebook; query a DC2 catalog and plot data; retrieve and display a DC2 image. |
| 02. Intermediate TAP Queries | Query, and retrieve DC2 catalog data with the Table Access Protocol (TAP) service. Use bokeh and holoviews to create interactive plots. |
| 03. Image Display and Manipulation | Display and manipulate DC2 images, explore image mask planes, create cutout and RGB images. |
| 04. Intro to Butler | Discover, query, retrieve, and display DC2 images and catalog data with the Generation 3 Butler. |
| 05. Intro to Source Detection | Use the LSST Science Pipelines tasks for image characterization, source detection, deblending, measurement, and to interact with a source footprint. |
| 06. Comparing Object and Truth Table | Retrieve and merge data from the DC2 Object and Truth-Match tables, and compare simulated and measured properties. |
| 07. *(time-domain tutorial)* | TBD |
| 08. *(data visualization)* | TBD |

These tutorials are subject to change, as the Rubin Science Platform and the LSST Science Pipelines are in active development.

The `prod` branch will appear in IDF RSP users' home directories.

Additional tutorials might be added as the [DP0 Delegates Assemblies](https://dp0-1.lsst.io/dp0-delegate-resources/index.html) series progresses.

More DP0 documentation can be found at [dp0-1.lsst.io](https://dp0-1.lsst.io).

**Acknowledgements**

These notebooks have drawn from these repositories:
 - https://github.com/lsst-sqre/notebook-demo
 - https://github.com/LSSTScienceCollaborations/StackClub

Many of the tutorial notebooks in this repository were originally developed by Stack Club members or Rubin staff, and have been altered and updated to be appropriate for DP0.
If these notebooks are used for a journal publication, please consider adding an acknowledgement that gives credit to the original creator(s) as listed in the notebook's header.

**Contributions**

Want to contribute a tutorial? Contact Melissa Graham via direct message at https://Community.lsst.org.

# RSP Workflow

How to use the RSP for DP0, including cloning notebooks, pushing changes to notebooks, etc.


## Main Documentation

The main documentation is at [https://dp0-1.lsst.io/](https://dp0-1.lsst.io/). The main [DP0 Delegate home page](https://dp0-1.lsst.io/dp0-delegate-resources/index.html) provides links to resources including the RSP instance link.

## RSP Instance

To launch the RSP interface, click [http://data.lsst.cloud/](http://data.lsst.cloud/).

## Workflow for GitHub hosted notebooks.

The [DP0 tutorial notebooks](https://github.com/rubin-dp0/tutorial-notebooks) are present on the RSP instances in `~/notebooks/tutorial-notebooks/`.

Change the git configuration if needed:

    $ git config --global user.email "brant@ucsc.edu"
    $ git config --global user.name "Brant Robertson"


Example workflow is to use the html cloning interface to GitHub:

    $ cd github
    $ git clone https://github.com/brantr/tutorial-notebooks.git
    $ [make changes to notebooks]
    $ git add [New notebook]
    $ git commit [Add messages]
    $ git push
    Username for 'https://github.com': brantr
    Password for 'https://brantr@github.com': 
    Enumerating objects: 4, done.
    Counting objects: 100% (4/4), done.
    Delta compression using up to 32 threads
    Compressing objects: 100% (3/3), done.
    Writing objects: 100% (3/3), 343 bytes | 343.00 KiB/s, done.
    Total 3 (delta 1), reused 0 (delta 0), pack-reused 0
    remote: Resolving deltas: 100% (1/1), completed with 1 local object.
    To https://github.com/brantr/tutorial-notebooks.git
        c196f81..ad6c67b  main -> main


To leave the RSP instance, do `File->Save All, Exit, and Logout`.

## AFW Documentation

Refer to the [AFW Documentation](https://pipelines.lsst.io/modules/lsst.afw.display/index.html) for information on the AFW datasets.