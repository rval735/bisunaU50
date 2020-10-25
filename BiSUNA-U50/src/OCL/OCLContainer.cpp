//
//  OCLContainer.cpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 18/8/19.
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

#include "OCLContainer.hpp"
#include "RandomUtils.hpp"
#include "OCL/Kernels/TycheI.h"
#include <climits>

// Make sure that the input pointer remains constant while the kernel executes
// that avoids the need to reallocate the argument when OCL executes the kernel
OCLContainer::OCLContainer(PConfig *pConf, UnifiedNeuralModel *unm, OCLRuntime *oclRuntime)
{
    // Create OCLRuntime
    enableProfiling = pConf->oclProfiling();
    devType = pConf->deviceType();
    string path = pConf->kernelFolder().c_str();
    string krnStateStr = pConf->kernelStateName();
    krnStateLcl = pConf->kernelStateUseLocalVars();
    string krnEpisodeStr = pConf->kernelEndEpisodeName();
    string krnEvolveStr = pConf->kernelEvolveName();
    vector<string> oclFiles = pConf->oclFiles();
    computeUnits = pConf->computeUnits();
    
    if (oclRuntime == NULL) {
        enableProfiling = pConf->oclProfiling();
        bool isMCU = computeUnits > 0;
        string flags = "-DCELL_POPULATION=" + to_string(unm->config.unmPopulation);
        flags += " -DNMSIZE=" + to_string(unm->config.unmMapSize);
        flags += " -DINPUT_SIZE=" + to_string(unm->config.unmObs);
        flags += " -DOUTPUT_SIZE=" + to_string(unm->config.unmActions);
        flags += " -DEPISODES_PER_AGENT=" + to_string(pConf->episodesPerAgent());
        isMCU ? flags += " -DMULTIPLECU" : flags;
        oclRt = new OCLRuntime(devType, path, oclFiles, flags, enableProfiling);
    }
    else {
        oclRt = oclRuntime;
    }
    
    // TODO!! make localWorkers an automatic variable that reads the ammount local memory
    // available in the device that can possibly fit in terms of CLUnifiedNeuralModel
    localWorkers = 1;
    
    kernelEpisodeEnabled = krnEpisodeStr.size() > 0;
    kernelEvolvedEnabled = krnEvolveStr.size() > 0;
    
    if (computeUnits < 0) {
        printf("Execution can only handle positive concurrent CU kernels\n");
        printf("Terminating execution.\n");
        exit(1);
    }
    else if (computeUnits > MAX_CU_KERNELS) {
            printf("Execution can only handle up-to %i concurrent CU kernels\n", MAX_CU_KERNELS);
            printf("Terminating execution.\n");
            exit(1);
    }
    else {
        uint idx = 0;
        do {
            kernelState[idx] = OpenCLUtils::loadKernel(oclRt->prg, krnStateStr.c_str());
            enableProfiling ? OpenCLUtils::displayKernelInfo(kernelState[idx], oclRt->device) : void();
            idx++;
        } while (idx < computeUnits);
    }
    
    if (kernelEpisodeEnabled) { kernelEpisode = OpenCLUtils::loadKernel(oclRt->prg, krnEpisodeStr.c_str()); }
    if (kernelEvolvedEnabled) { kernelEvolve = OpenCLUtils::loadKernel(oclRt->prg, krnEvolveStr.c_str()); }
    enableProfiling && kernelEpisodeEnabled ? OpenCLUtils::displayKernelInfo(kernelEpisode, oclRt->device) : void();
    enableProfiling && kernelEvolvedEnabled ? OpenCLUtils::displayKernelInfo(kernelEvolve, oclRt->device) : void();
    
    // Create IO Buffers
    ushortT population = unm->config.unmPopulation;
    ushortT inputLength = unm->config.unmObs;
    ushortT inputSize = inputLength * population;
    ushortT outputLength = unm->config.unmActions;
    ushortT outputSize = outputLength * population;
    
    inBuffSize = sizeof(ParameterType) * inputSize;
    inBuffer = OpenCLUtils::clBufferCreation(oclRt->ctx, inBuffSize, CL_MEM_READ_ONLY);
    
    outBuffSize = sizeof(ParameterType) * outputSize;
    outBuffer = OpenCLUtils::clBufferCreation(oclRt->ctx, outBuffSize, CL_MEM_WRITE_ONLY);
    
    CLUnifiedNeuralModel localUNM = OCLBridge::toCLUNM(*unm);
    size_t clUNMSize = sizeof(CLUnifiedNeuralModel);
    unmBuffer = OpenCLUtils::clBufferCreation(oclRt->ctx, clUNMSize, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &localUNM);
    
    // Only load TycheIBuffer when using local state kernel, given that processState and spectrumDiversityEvolve
    // requires tycheI objects.
    if (krnStateLcl == true) {
        rewardsBuffSize = sizeof(float) * population;
        rewardsBuffer = OpenCLUtils::clBufferCreation(oclRt->ctx, rewardsBuffSize, CL_MEM_READ_ONLY);
        
        tycheIBuffer = randomizeTycheI(oclRt->ctx, population);
    }
    
    ushortT workItems = CELL_POPULATION;
    int offset = 0;
    int mcuCounter = 0;
    uint splitWork = computeUnits > 0 ? workItems / computeUnits : 0;
    clUNM = &localUNM;
    do {
        loadParametersKrnState(mcuCounter, offset, offset + splitWork);
        mcuCounter++;
        offset += splitWork;
    } while (mcuCounter < computeUnits);

    inRef = (ParameterType *)OpenCLUtils::clBufferMap(oclRt->queue, inBuffer, inBuffSize, CL_MAP_WRITE);
    outRef = (ParameterType *)OpenCLUtils::clBufferMap(oclRt->queue, outBuffer, outBuffSize, CL_MAP_READ);
    clUNM = (CLUnifiedNeuralModel *)OpenCLUtils::clBufferMap(oclRt->queue, unmBuffer, clUNMSize, CL_MAP_READ | CL_MAP_WRITE);
    // The use of a subbuffer here helps to identify which part of the whole "unmBuffer" will be used to
    // be passed to a kernel which only uses CLCells, most likely a "processStateG" kernel.
    size_t clCellSize = sizeof(clUNM->cells);
    // NOTE!! This is just a pointer to the first part of the unmBuffer, in case there are problems with the
    // memory bounds about this, it should be considered to use the whole unmBuffer instead inside the Kernel
    cellBufferMap = (CLCell *)OpenCLUtils::clBufferMap(oclRt->queue, unmBuffer, clCellSize, CL_MAP_READ | CL_MAP_WRITE);
    
    if (krnStateLcl == true) {
        rewardsRef = (float *)OpenCLUtils::clBufferMap(oclRt->queue, rewardsBuffer, rewardsBuffSize, CL_MAP_WRITE);
    }
    else {
        rewardsRef = new float[population];
    }
}

OCLContainer::~OCLContainer()
{
    oclRt->queue.enqueueUnmapMemObject(inBuffer, inRef);
    oclRt->queue.enqueueUnmapMemObject(outBuffer, outRef);
    oclRt->queue.enqueueUnmapMemObject(unmBuffer, cellBufferMap);
    oclRt->queue.enqueueUnmapMemObject(unmBuffer, clUNM);
    oclRt->queue.enqueueUnmapMemObject(rewardsBuffer, rewardsRef);
    
    if (krnStateLcl == false) {
        delete [] rewardsRef;
    }
    
    if (oclRt != NULL) {
        delete oclRt;
    }
}

void CL_CALLBACK clWaitCallback(cl_event e, cl_int s, void *d) {
    bool *data = (bool *)d;
    *data = false;
}

void OCLContainer::exeOCL(cl::Kernel kernel, function<void(double diffTask)> profFn)
{
    cl_int err;
    size_t workUnits = CELL_POPULATION;
    cl::Event clTask;
    
    if (computeUnits == 0) {
        /* Enqueue the command queue to the device */
//        cl::NDRange workRange(workUnits);
        err = oclRt->queue.enqueueNDRangeKernel(kernel, cl::NullRange, workUnits, cl::NullRange, NULL, &clTask);
        oclCheckError(err, "Couldn't enqueue the kernel execution command");
    }
    else {
        err = oclRt->queue.enqueueTask(kernel, NULL, &clTask);
        oclCheckError(err, "Couldn't call kernel with execution command enqueue task");
    }
    
    if (enableProfiling) {
        if (profFn == NULL) {
            OpenCLUtils::printProfiling("exeOCL", clTask);
        }
        else {
            double tsk = OpenCLUtils::printProfiling(NULL, clTask);
            profFn(tsk);
        }
        
    }
    
//// If it is necessary to wait for the kernel to finish, uncomment one of the following sections of code,
//// with preference over the first with clFinish
//    err = clFinish(oclRt->queue);
//    oclCheckError(err, "Error at exeOCL while waiting for queue to finish\n");
//--------<
//    if (waitToFinish) {
//        bool notFinished = true;
//        err = clSetEventCallback(clTask, CL_COMPLETE, clWaitCallback, &notFinished);
//        oclCheckError(err, "Couldn't set callback event");
//        while (notFinished == true) {
//            continue;
//        }
//    }
}

void OCLContainer::loadParametersKrnStateGbl(uint kernelIndex, int start, int end)
{
    vector<cl::Buffer>clElems = {inBuffer, outBuffer, unmBuffer};
    
    OpenCLUtils::loadArgs(clElems, kernelState[kernelIndex]);
    
    if (start != end) {
        cl_uint argIdx = (cl_uint)clElems.size();
        cl_int err = kernelState[kernelIndex].setArg(argIdx, start);
        oclCheckError(err, "Error setting kernel start index parameter");
        err = kernelState[kernelIndex].setArg(argIdx + 1, end);
        oclCheckError(err, "Error setting kernel end index parameter");
    }
}

void OCLContainer::loadParametersKrnState(uint kernelIndex, int start, int end)
{
    // Check here if the kernel has set a "local" only processing, most likely related
    // to "processState" that has local variables inside. When it is false, it will
    // try to load the kernel "processStateG" which uses only global parameters
    if (krnStateLcl == false) {
        return loadParametersKrnStateGbl(kernelIndex, start, end);
    }
    
    vector<cl::Buffer>clElems;
    clElems.reserve(4);
    
    clElems.push_back(inBuffer);
    clElems.push_back(outBuffer);
    clElems.push_back(tycheIBuffer);
    clElems.push_back(unmBuffer);
    
    cl::Kernel knlState = kernelState[kernelIndex];
    OpenCLUtils::loadArgs(clElems, knlState);
    
    cl_int err;
    ushortT maxConns = 0;
    ushortT maxNeurons = 0;

    for (uintT i = 0; i < CELL_POPULATION; i++) {
        CLCell cell = clUNM->cells[i];
        if (cell.nrsSize > maxNeurons) {
            maxNeurons = cell.nrsSize;
        }

        if (cell.connSize > maxConns) {
            maxConns = cell.connSize;
        }
    }

    uintT clElemsSize = (uintT)clElems.size();
    cl::LocalSpaceArg arg = cl::__local(maxNeurons * sizeof(CLNeuron) * localWorkers);
    err = knlState.setArg(clElemsSize, arg);
    oclCheckError(err, "Error setting kernel parameter CLNeuron");
    arg = cl::__local(maxConns * sizeof(CLConnection) * localWorkers);
    err = knlState.setArg(clElemsSize + 1, arg);
    oclCheckError(err, "Error setting kernel parameter CLConnection");
    arg = cl::__local(INPUT_SIZE * sizeof(ParameterType) * localWorkers);
    err = knlState.setArg(clElemsSize + 2, arg);
    oclCheckError(err, "Error setting kernel parameter local input");
    arg = cl::__local(OUTPUT_SIZE * sizeof(ParameterType) * localWorkers);
    err = knlState.setArg(clElemsSize + 3, arg);
    oclCheckError(err, "Error setting kernel parameter local output");
}

void OCLContainer::loadParametersKrnEpisode()
{
    vector<cl::Buffer>clElems = {rewardsBuffer, unmBuffer};
    OpenCLUtils::loadArgs(clElems, kernelEpisode);
}

void OCLContainer::loadParametersKrnEvolve()
{
    vector<cl::Buffer>clElems = {tycheIBuffer, unmBuffer};
    OpenCLUtils::loadArgs(clElems, kernelEvolve);
}

cl::Buffer OCLContainer::randomizeTycheI(cl::Context ctx, size_t population)
{
    tycheIState tycheIArr[population];
    for (uintT i = 0; i < population; i++) {
        tycheIState &tSt = tycheIArr[i];
        tSt.a = RandomUtils::randomUInt(UINT_MAX - 1);
        tSt.b = RandomUtils::randomUInt(UINT_MAX - 1);
        tSt.c = RandomUtils::randomUInt(UINT_MAX - 1);
        tSt.d = RandomUtils::randomUInt(UINT_MAX - 1);
    }

    size_t tycheIBuffSize = sizeof(tycheIState) * population;
    cl::Buffer buffer = OpenCLUtils::clBufferCreation(ctx, tycheIBuffSize, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &tycheIArr);
    return buffer;
}
