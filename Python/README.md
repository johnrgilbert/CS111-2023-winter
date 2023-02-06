## Setting up and installing Anaconda, Jupyter, and Python

We will be using [Python](https://docs.python.org/3/) (version 3.5 or higher) for programming in this course, leaning heavily on the following three packages:
- [numpy](https://numpy.org/doc/stable/): Numerical computing with arrays and matrices
- [scipy](https://docs.scipy.org/doc/scipy/reference/): More advanced numerical computing, including sparse matrices
- [matplotlib](https://matplotlib.org/stable/contents.html): Plotting and visualization

I strongly recommend that you set up your own laptop or computer
to run Jupyter and Python (and numpy, scipy, and matplotlib).
That's the way I do it myself. The TAs will demo the setup process
in the first section, on Wednesday, January 11.

We will use a handful of demo routines in Python that live in a
module called **cs111**. You'll need to be able to import them too.

### Clone the CS 111 directory from GitHub

The first thing to do is to use **git** to clone the **CS111-2023-winter** tree from the [CS 111 GitHub site](https://github.com/johnrgilbert/CS111-2023-winter/tree/main).
Navigate to the site, click on the green "Code" button, and select
a method (SSH, HTTPS, or GitHub CLI) to use. I prefer SSH myself
because you don't have to type passwords all the time, but do
whatever works for you. In my old-fashioned way, I would just copy
the string under SSH, then on my own laptop I would get a command
line and navigate to wherever I wanted the **CS111-2023-winter/**
directory to be, and say **git clone the-string**.
This gets you a local copy of all the lecture materials, reading materials,
homework, syllabus, and so forth as well as the **cs111** Python module.

Any time you want to make sure your copy is up to date,
you can just navigate to your **CS111-2023-winter/** directory
and say **git pull**. 

### Download Anaconda 3

All our software -- python, numpy, scipy, matplotlib -- is
part of the free Anaconda distribution, except for the **cs111** module.
Go to the [Anaconda download page](https://www.anaconda.com/products/individual#Downloads), and download the individual edition for whatever OS your
computer uses. (I use the MacOS version on my MacBook laptop, and I've
also used the Linux version, and there's a version for Windows but I
have not used it.) Follow the instructions it gives you to install it 
on your computer. With any luck, this will make everything we need
seamlessly available.

Someone once asked me if they could install Anaconda and the course 
software on an iPad. I haven't tried. The best answer I found online 
is "Nothing is impossible but it is impossible."

### Run Python in a Jupyter notebook

You should be able to just navigate to the directory you want to work in
on your computer and say **jupyter notebook**. 
That will open a page in your web browser pointing to your current 
directory, and from there you can start individual notebook files.
The first time, you might want to copy into that directory the file 
**CS111-2023-winter/Python/Scratch.ipynb**, which is a notebook with all
the standard imports for CS111.
You can run the notebook files from lecture the same way.
