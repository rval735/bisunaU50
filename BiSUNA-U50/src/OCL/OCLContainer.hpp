//
//  OCLContainer.hpp
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

#ifndef OCLContainer_hpp
#define OCLContainer_hpp

#include "OCLBridge.hpp"

template <typename T>
void readOrWriteCLBuffer(cl::CommandQueue queue, cl::Buffer rwBuffer, size_t rwSize, T *rwOutput,
                         bool readBuffer = true,
                         bool profiling = false,
                         function<void(double diffTask)> profFn = NULL)
{
    cl_int err;
    size_t buffSize = rwSize * sizeof(T);
    if (profiling) {
        cl::Event profRW;
        if (readBuffer == true) {
            err = queue.enqueueReadBuffer(rwBuffer, CL_TRUE, 0, buffSize, NULL, NULL, &profRW);
        }
        else {
            err = queue.enqueueWriteBuffer(rwBuffer, CL_TRUE, 0, buffSize, rwOutput, NULL, &profRW);
        }
        
        oclCheckError(err, "Couldn't enqueue the read/write buffer output command with profile");

        if (profFn == NULL) {
            OpenCLUtils::printProfiling("rwBufferIndex", profRW);
        }
        else {
            double opt = OpenCLUtils::printProfiling(NULL, profRW);
            profFn(opt);
        }
    }
    else {
        if (readBuffer == true) {
            err = queue.enqueueReadBuffer(rwBuffer, CL_TRUE, 0, buffSize, rwOutput);
        }
        else {
            err = queue.enqueueWriteBuffer(rwBuffer, CL_TRUE, 0, buffSize, rwOutput);
        }
        oclCheckError(err, "Couldn't enqueue the read/write buffer output command");
    }
}

struct OCLContainer
{
    static const uint MAX_CU_KERNELS = 16;
    CLUnifiedNeuralModel *clUNM = NULL;
    OCLRuntime *oclRt = NULL;
    cl_device_type devType = CL_DEVICE_TYPE_CPU;
    cl::Kernel kernelState[MAX_CU_KERNELS];// = NULL;
    cl::Kernel kernelEpisode;
    bool kernelEpisodeEnabled = false;
    cl::Kernel kernelEvolve;
    bool kernelEvolvedEnabled = false;
    cl::Buffer unmBuffer;
    CLCell *cellBufferMap = NULL;
    cl::Buffer tycheIBuffer;
    ParameterType *inRef = NULL;
    cl::Buffer inBuffer;
    ushortT inBuffSize = 0;
    ParameterType *outRef = NULL;
    cl::Buffer outBuffer;
    ushortT outBuffSize = 0;
    float *rewardsRef = NULL;
    cl::Buffer rewardsBuffer;
    ushortT rewardsBuffSize = 0;
    bool enableProfiling = false;
    size_t localWorkers = 1;
    bool krnStateLcl = true;
    long computeUnits = 0;
    
    OCLContainer(PConfig *pConf, UnifiedNeuralModel *unm, OCLRuntime *oclRuntime);
    ~OCLContainer();
    
    void exeOCL(cl::Kernel kernel, function<void(double diffTask)> profFn = NULL);
    void loadParametersKrnState(uint kernelIndex = 0, int start = -1, int end = -1);
    void loadParametersKrnEpisode();
    void loadParametersKrnEvolve();
    
    static cl::Buffer randomizeTycheI(cl::Context ctx, size_t population);
    
private:
    void loadParametersKrnStateGbl(uint kernelIndex = 0, int start = -1, int end = -1);
};

#endif /* OCLContainer_hpp */
