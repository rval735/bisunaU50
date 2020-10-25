# BiSUNA - OpenCL

**Bi**nary **S**pectrum-unified **N**euroevolutionary **A** rchitecture capable of executing OpenCL
kernels. This project presents a different approach to train from scratch
Binary Neural Networks (BNN) using neuroevolution as its base technique (gradient
descent free) that takes advantage of OpenCL acceleration.

Obtain non-linear Deep NN that use binary values in weights, activations, operations
and is completely gradient free; which brings us to the brief summary of the
capabilities of shown with this implementation:

• Weights and activations are represented with fixed length bitset (16 bits)
• Only logic operations (AND, XOR, OR...) are used, no need of Arithmetic Logic Unit (ALU)
• Neuroevolution is employed to drive the space search
• BNN do not have a fixed network topology, which adapt and optimize for the problem at hand.

![Float - Binary neural network](resources/Float-BinaryNetwork.png)

## Installation

This project has been tested with correct execution on Linux and MacOS. It has two main library
requirement in order to be able to compile code from the makefile. The first, it needs the
OpenCL development libraries. If running in a Debian/Ubuntu distribution it is as simple
as running

```
	sudo apt install build-essential opencl-dev
```

The second is GRPC, which involves more steps necessary to compile the code. In order to
recreate that environment, check the file "[Install-Compile-Execute.sh](scripts/InstallGRPC.sh)"

If an FPGA is the target platform, an special compiler is needed to obtain the bitstream
used to deploy on these boards, that is used to reprogram it at time the binary executes.

## Execution

Once a binary is obtained, it is just about to executing the file along with a configuration
file that provides all runtime details like the following:

```
./bisuna resources/BiSUNAConf.ini
```

The ini file will instruct BiSUNA what type of execution, device type and all other details.
Have in consideration that BiSUNAConf.ini has a parameter "EnvironmentConf", which should
have the path to another "ini" file with the correct environment details.

## Reinfocement Learning Environments

- Mountain Car

Note: This project was based on repository from [Physis-Shard](https://github.com/zweifel/Physis-Shard) implementation.

## FPGA Resource Utilization

Below are shown three screenshots of the resource utilization when BiSUNA was compiled for the Cyclone V:

![FPGA Fitter](resources/utilization/FPGA-Fitter.png)

![FPGA Maximum Frequency](resources/utilization/FPGA-FMax.png)

![FPGA Power Dissipation](resources/utilization/FPGA-Power.png)


If you would like to reference this work, you can use the following [bibtex](papers/APCCAS2019.bib):

```
@inproceedings{8953134,
	Abstract = {With the explosive interest in the utilization of Neural Networks, several approaches have taken place to make them faster, more accurate or power efficient; one technique used to simplify inference models is the utilization of binary representations for weights, activations, inputs and outputs. This paper presents a novel approach to train from scratch Binary Neural Networks using neuroevolution as its base technique (gradient descent free), to then apply such results to standard Reinforcement Learning environments tested in the OpenAI Gym. The results and code can be found in https://github.com/rval735/BiSUNA.},
	Author = {R. {Valencia} and C. {Sham} and O. {Sinnen}},
	Booktitle = {2019 IEEE Asia Pacific Conference on Circuits and Systems (APCCAS)},
	Date-Added = {2020-01-20 16:32:29 +1300},
	Date-Modified = {2020-01-20 16:32:40 +1300},
	Doi = {10.1109/APCCAS47518.2019.8953134},
	Issn = {null},
	Keywords = {BNN; Neuroevolution;binary neural networks;BiSUNA;discrete optimization},
	Month = {Nov},
	Pages = {301-304},
	Title = {Using Neuroevolved Binary Neural Networks to solve reinforcement learning environments},
	Year = {2019},
	Bdsk-Url-1 = {https://doi.org/10.1109/APCCAS47518.2019.8953134}}

```
Of course, if you would like to get the paper, it is located here: [APCCAS 2019 Paper](papers/APCCAS2019.pdf)

Second paper, this makes [reference](papers/FPT2019.bib) to the FPGA execution of BiSUNA [FPT 2019 Paper](papers/FPT2019.pdf):
```
@inproceedings{8977877,
	Abstract = {The exponential progress of semiconductor tech-nologies has enabled the proliferation of deep learning as a prominent area of research, where neural networks have demon-strated its effectiveness to solve very hard multi dimensional problems. This paper focuses on one in particular, Binary Neural Networks (BNN), which use fixed length bits in its connections and logic functions to perform excitation operations. Exploiting those characteristics, hardware accelerators that integrate field-programmable gate arrays (FPGAs) have been adopted to hasten inference of deep learning networks, given its proficiency to maximize parallelism and energy efficiency. This work will show how the algorithm Binary Spectrum-diverse Unified Neuroevolution Architecture (BiSUNA) can perform training and inference on FPGA without the need of gradient descent. Source code can be found in github.com/rval735/bisunaocl},
	Author = {R. {Valencia} and C. W. {Sham} and O. {Sinnen}},
	Booktitle = {2019 International Conference on Field-Programmable Technology (ICFPT)},
	Date-Added = {2020-02-04 20:40:00 +1300},
	Date-Modified = {2020-02-04 20:40:39 +1300},
	Doi = {10.1109/ICFPT47387.2019.00076},
	Issn = {null},
	Keywords = {BNN, FPGA;BiSUNA;Binary Neural Network;TWEANN;Evolutionary Algorithm},
	Month = {Dec},
	Pages = {395-398},
	Title = {Evolved Binary Neural Networks Through Harnessing FPGA Capabilities},
	Year = {2019},
	Bdsk-Url-1 = {https://doi.org/10.1109/ICFPT47387.2019.00076}}
```

To check chosen plaintext attack, check the CPA branch.
