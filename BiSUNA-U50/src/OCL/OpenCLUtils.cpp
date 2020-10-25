//
//  OpenCLUtils.cpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 15/5/19.
//  Copyright © 2019 R. All rights reserved.
//

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "OpenCLUtils.hpp"
#include <cstdarg>
#include <fstream>

OCLRuntime::OCLRuntime(cl_device_type devType, const string &folder, const vector<string> &clFiles, string extraFlags, bool profileDevice)
{
	cl_int err = CL_SUCCESS;
    vector<cl::Platform> platforms;
    vector<cl::Device> pltDevices, ctxDevices;
    string deviceName;
    err = cl::Platform::get(&platforms);
    oclCheckError(err, "Couldn't read platform from OCL");
    int devIdx = -1;
    int pltIdx = -1;
    for (cl::Platform plt : platforms) {
        /* Access all devices within a platform */
    	err = plt.getDevices(CL_DEVICE_TYPE_ALL, &pltDevices);
    	cl::Context ctx(pltDevices);
    	ctxDevices = ctx.getInfo<CL_CONTEXT_DEVICES>();

        for (cl::Device ctxDev : ctxDevices) {
        	deviceName = ctxDev.getInfo<CL_DEVICE_NAME>();
        	cl_device_type deviceType = ctxDev.getInfo<CL_DEVICE_TYPE>();
            profileDevice ? OpenCLUtils::displayCLDevice(ctxDev) : void();
            
            if (devType == deviceType) {
            	OpenCLUtils::displayCLDevice(ctxDev);
            	device = ctxDev;
                devIdx++;
                break;
            }

            devIdx++;
        }

        pltIdx++;
    }
    
    if (devIdx == -1) {
        printf("No OpenCL device of the requested type could be found in this system. Change configuration and run again\n");
        exit(1);
    }
    
    /* Create the context */
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[pltIdx])(), 0};
    ctx = cl::Context(device, props, NULL, NULL, &err);
    oclCheckError(err, "Couldn't create a context");
    
    /* Create a CL command queue for the device*/
    uint enableProfiling = profileDevice ? CL_QUEUE_PROFILING_ENABLE : 0;
    
    if (devType == CL_DEVICE_TYPE_ACCELERATOR) {
    	queue = cl::CommandQueue(ctx, device, enableProfiling | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        oclCheckError(err, "Couldn't create the command queue");
        prg = OpenCLUtils::readBinProgram(folder, clFiles, ctx, device);
    }
    else {
    	queue = cl::CommandQueue(ctx, device, enableProfiling, &err);
        oclCheckError(err, "Couldn't create the command queue");
        prg = OpenCLUtils::compileProgram(folder, clFiles, ctx, device, extraFlags);
    }
}

OCLRuntime::~OCLRuntime()
{
}

// Interesting example: https://gist.github.com/tzutalin/51a821f15a735024f16b
void OpenCLUtils::displayCLDevice(cl::Device device)
{
    printf(" ----------------------------------\n");
    
    string deviceName = device.getInfo<CL_DEVICE_NAME>();
    printf("  CL_DEVICE_NAME:\t\t\t\t\t%s\n", deviceName.c_str());
    
    string version = device.getInfo<CL_DEVICE_VERSION>();
    printf("  CL_DEVICE_VERSION:\t\t\t\t%s\n", version.c_str());
    
    cl_uint cu = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    printf("  CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%u\n", cu);
    
    size_t workitem_dims = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
    printf("  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:%ld\n", workitem_dims);
    
    vector<size_t> workitem_size = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    printf("  CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%ld / %ld / %ld \n", workitem_size[0], workitem_size[1], workitem_size[2]);

    size_t workgroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    printf("  CL_DEVICE_MAX_WORK_GROUP_SIZE:\t%ld\n", workgroupSize);

    cl_ulong max_mem_alloc_size = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    printf("  CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t%u MByte\n", (unsigned int)(max_mem_alloc_size / (1024 * 1024)));

    cl_ulong memSize = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    printf("  CL_DEVICE_GLOBAL_MEM_SIZE:\t\t%u MByte\n", (unsigned int)(memSize / (1024 * 1024)));

    memSize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    printf("  CL_DEVICE_LOCAL_MEM_SIZE:\t\t\t%u KByte\n", (unsigned int)(memSize / 1024));
    
    memSize = device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
    printf("  CL_DEVICE_MEM_BASE_ADDR_ALIGN:\t%u KByte\n", (unsigned int)(memSize / 1024));
    
    size_t timeRes = device.getInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION>();
    printf("  CL_DEVICE_PROFILING_TIMER_RESOLUTION:\t%lu ns\n", timeRes);
    
    printf(" ----------------------------------\n");
}

void OpenCLUtils::displayKernelInfo(cl::Kernel kernel, cl::Device device)
{
    printf(" ----------------------------------\n");
    
    string name = kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
    printf("  CL_KERNEL_FUNCTION_NAME:\t\t\t%s\n", name.c_str());
    
    size_t numArgs = kernel.getInfo<CL_KERNEL_NUM_ARGS>();
    printf("  CL_KERNEL_NUM_ARGS:\t\t\t\t%ld\n", numArgs);
        
    size_t wgSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
    printf("  CL_KERNEL_WORK_GROUP_SIZE:\t\t%ld\n", wgSize);
    
    auto cwgSize = kernel.getWorkGroupInfo<CL_KERNEL_COMPILE_WORK_GROUP_SIZE>(device);
    printf("  CL_KERNEL_COMPILE_WORK_GROUP_SIZE:%ld, %ld, %ld\n", cwgSize[0], cwgSize[1], cwgSize[2]);
    
    size_t wgMultiple = kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
    printf("  CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:%ld\n", wgMultiple);
    
//    auto globalUsage = kernel.getWorkGroupInfo<CL_KERNEL_GLOBAL_WORK_SIZE>(device);
//    printf("  CL_KERNEL_GLOBAL_WORK_SIZE:\t\t\t%lld\n", globalUsage);
    
    cl_ulong localUsage = kernel.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device);
    printf("  CL_KERNEL_LOCAL_MEM_SIZE:\t\t\t%lld\n", localUsage);
    
    cl_ulong privUsage = kernel.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(device);
    printf("  CL_KERNEL_PRIVATE_MEM_SIZE:\t\t%lld\n", privUsage);
    
    printf(" ----------------------------------\n");
}

cl::Program OpenCLUtils::readBinProgram(const string &folder, vector<string>clFiles, cl::Context ctx, cl::Device device)
{
    if (clFiles.size() > 1) {
        printf("Only one binary file is supported. Finishing execution");
        exit(1);
    }
    
    string fwn = folder;
    fwn.append(clFiles[0]);
    auto res = loadFile(fwn.c_str(), true);
    cl_int err = CL_SUCCESS;
    vector<cl_int> binSt;
    cl::Program::Binaries binFile = {{(void *)res.first, res.second}};
    cl::Program pgr(ctx, {device}, binFile, &binSt, &err);

    oclCheckError(err, "Failed to create program with binary file");
    oclCheckError(binSt[0], "Failed to load binary for device");
    
    delete [] res.first;
    
//    err = prg.build()
//    err = clBuildProgram(pgr, 0, NULL, "", NULL, NULL);
//    oclCheckError(err, "Failed to build binary program");
    
    return pgr;
}

/* Read program file and place content into buffer */
cl::Program OpenCLUtils::compileProgram(const string &folder, vector<string>clFiles, cl::Context ctx, cl::Device device, string extraFlags)
{
    cl_int err;
    string programString;
    for (string clName : clFiles) {
        ifstream programFile(folder + clName);
        string source(std::istreambuf_iterator<char>(programFile), (std::istreambuf_iterator<char>()));
        programString += source;
    }
    
    /* Create program from file */
    cl::Program program(ctx, programString, false, &err);
    oclCheckError(err, "Couldn't create the program");
    
    /* Build program */
    string folderFlag = "-I" + folder + " ";
    string flags = "-cl-finite-math-only -cl-no-signed-zeros ";
    
#ifdef CONTINUOUS_PARAM
    extraFlags += " -DCONTINUOUS_PARAM";
#endif
    
#ifdef UNIT_TEST
    extraFlags += " -DPRECISION";
#endif
    
    string options = flags + folderFlag + extraFlags;
    program.build({device}, options.c_str());
    cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
    if (status == CL_BUILD_ERROR) {
        // Get the build log
        string name = device.getInfo<CL_DEVICE_NAME>();
        string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        printf("Build log for %s:\n%s\n", name.c_str(), buildlog.c_str());
        exit(1);
    }
    
    return program;
}

cl::Kernel OpenCLUtils::loadKernel(cl::Program prg, const char *krlName)
{
    cl_int err;
    cl::Kernel kernel(prg, krlName, &err);
    oclCheckError(err, krlName);
    return kernel;
}

void OpenCLUtils::loadArgs(const vector<cl::Buffer> &elements, cl::Kernel kernel)
{
    cl_int err;
    size_t elemSize = elements.size();
    string errMsg = "Couldn't set the kernel argument #";
    
    for (cl_uint i = 0; i < elemSize; i++) {
        err = kernel.setArg(i, elements[i]);
        oclCheckError(err, (errMsg + to_string(i)).c_str());
    }
}

void OpenCLUtils::checkError(int line, const char *file, cl_int error, const char *msg, ...) {
    // If not successful
    if(error != CL_SUCCESS) {
        // Print line and file
        printf("ERROR: ");
        oclPrintError(error);
        printf("\nLocation: %s:%d\n", file, line);
        
        // Print custom message.
        va_list vl;
        va_start(vl, msg);
        vprintf(msg, vl);
        printf("\n");
        va_end(vl);
        
        exit(error);
    }
}

// Print the error associciated with an error code
void OpenCLUtils::oclPrintError(cl_int error)
{
    // Print error message
    switch(error)
    {
        case -1:
            printf("CL_DEVICE_NOT_FOUND ");
            break;
        case -2:
            printf("CL_DEVICE_NOT_AVAILABLE ");
            break;
        case -3:
            printf("CL_COMPILER_NOT_AVAILABLE ");
            break;
        case -4:
            printf("CL_MEM_OBJECT_ALLOCATION_FAILURE ");
            break;
        case -5:
            printf("CL_OUT_OF_RESOURCES ");
            break;
        case -6:
            printf("CL_OUT_OF_HOST_MEMORY ");
            break;
        case -7:
            printf("CL_PROFILING_INFO_NOT_AVAILABLE ");
            break;
        case -8:
            printf("CL_MEM_COPY_OVERLAP ");
            break;
        case -9:
            printf("CL_IMAGE_FORMAT_MISMATCH ");
            break;
        case -10:
            printf("CL_IMAGE_FORMAT_NOT_SUPPORTED ");
            break;
        case -11:
            printf("CL_BUILD_PROGRAM_FAILURE ");
            break;
        case -12:
            printf("CL_MAP_FAILURE ");
            break;
        case -13:
            printf("CL_MISALIGNED_SUB_BUFFER_OFFSET ");
            break;
        case -14:
            printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ");
            break;
        case -30:
            printf("CL_INVALID_VALUE ");
            break;
        case -31:
            printf("CL_INVALID_DEVICE_TYPE ");
            break;
        case -32:
            printf("CL_INVALID_PLATFORM ");
            break;
        case -33:
            printf("CL_INVALID_DEVICE ");
            break;
        case -34:
            printf("CL_INVALID_CONTEXT ");
            break;
        case -35:
            printf("CL_INVALID_QUEUE_PROPERTIES ");
            break;
        case -36:
            printf("CL_INVALID_COMMAND_QUEUE ");
            break;
        case -37:
            printf("CL_INVALID_HOST_PTR ");
            break;
        case -38:
            printf("CL_INVALID_MEM_OBJECT ");
            break;
        case -39:
            printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ");
            break;
        case -40:
            printf("CL_INVALID_IMAGE_SIZE ");
            break;
        case -41:
            printf("CL_INVALID_SAMPLER ");
            break;
        case -42:
            printf("CL_INVALID_BINARY ");
            break;
        case -43:
            printf("CL_INVALID_BUILD_OPTIONS ");
            break;
        case -44:
            printf("CL_INVALID_PROGRAM ");
            break;
        case -45:
            printf("CL_INVALID_PROGRAM_EXECUTABLE ");
            break;
        case -46:
            printf("CL_INVALID_KERNEL_NAME ");
            break;
        case -47:
            printf("CL_INVALID_KERNEL_DEFINITION ");
            break;
        case -48:
            printf("CL_INVALID_KERNEL ");
            break;
        case -49:
            printf("CL_INVALID_ARG_INDEX ");
            break;
        case -50:
            printf("CL_INVALID_ARG_VALUE ");
            break;
        case -51:
            printf("CL_INVALID_ARG_SIZE ");
            break;
        case -52:
            printf("CL_INVALID_KERNEL_ARGS ");
            break;
        case -53:
            printf("CL_INVALID_WORK_DIMENSION ");
            break;
        case -54:
            printf("CL_INVALID_WORK_GROUP_SIZE ");
            break;
        case -55:
            printf("CL_INVALID_WORK_ITEM_SIZE ");
            break;
        case -56:
            printf("CL_INVALID_GLOBAL_OFFSET ");
            break;
        case -57:
            printf("CL_INVALID_EVENT_WAIT_LIST ");
            break;
        case -58:
            printf("CL_INVALID_EVENT ");
            break;
        case -59:
            printf("CL_INVALID_OPERATION ");
            break;
        case -60:
            printf("CL_INVALID_GL_OBJECT ");
            break;
        case -61:
            printf("CL_INVALID_BUFFER_SIZE ");
            break;
        case -62:
            printf("CL_INVALID_MIP_LEVEL ");
            break;
        case -63:
            printf("CL_INVALID_GLOBAL_WORK_SIZE ");
            break;
        default:
            printf("UNRECOGNIZED ERROR CODE (%d)", error);
    }
}

// Loads a file in binary form.
pair<unsigned char *, size_t> OpenCLUtils::loadFile(const char *filePath, bool binary)
{
    // Open the File
    FILE *fp;
    size_t fSize = 0;
    string readMode = binary ? "rb" : "r";
    fp = fopen(filePath, readMode.c_str());
    
    if(fp == 0) {
        printf("File %s could not be read, failing execution.\n", filePath);
        exit(1);
    }
    
    // Get the size of the file
    fseek(fp, 0, SEEK_END);
    fSize = ftell(fp);
    
    // Allocate space for the binary
    unsigned char *buffer = new unsigned char[fSize];
    
    // Go back to the file start
    rewind(fp);
    
    // Read the file into the binary
    if(fread((void*)buffer, fSize, 1, fp) == 0) {
        delete[] buffer;
        fclose(fp);
        printf("Buffer of file %s with size %ld could not be read, failing execution.\n", filePath, fSize);
        exit(1);
    }
    
    fclose(fp);
    
    return make_pair(buffer, fSize);
}

cl_ulong OpenCLUtils::printProfiling(const char *name, cl::Event profEvent)
{
    cl_ulong timeStart = profEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong timeEnd = profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    cl_ulong diffTime = (timeEnd - timeStart);
    
    if (name != NULL) {
        printf("Profiling %s: %llu ns \n", name, diffTime);
    }
    
    return diffTime;
}

cl::Buffer OpenCLUtils::clBufferCreation(cl::Context ctx, size_t memSize, cl_mem_flags flags, void *dta)
{
    cl_int err;
    cl::Buffer res(ctx, flags, memSize, dta, &err);
    oclCheckError(err, "Couldn't create a buffer object");
    return res;
}

cl::Buffer OpenCLUtils::clSubBufferCreation(cl::Buffer mainBff, size_t memOrigin, size_t memSize, cl_mem_flags flags)
{
    cl_int err;
    size_t config[2] = {memOrigin, memSize};
    cl::Buffer res = mainBff.createSubBuffer(flags, CL_BUFFER_CREATE_TYPE_REGION, (void *)config, &err);
    oclCheckError(err, "Couldn't create a sub buffer object");
    return res;
}

void *OpenCLUtils::clBufferMap(cl::CommandQueue clQ, cl::Buffer memPtr, size_t memSize, cl_map_flags flags)
{
    cl_int err;
    return clQ.enqueueMapBuffer(memPtr, CL_TRUE, flags, 0, memSize, NULL, NULL, &err);
    oclCheckError(err, "Couldn't create a map buffer object");
//    queue.enqueueMapBuffer(bufferB, CL_TRUE, CL_MAP_READ, 0, sizeof(dataB));

}
