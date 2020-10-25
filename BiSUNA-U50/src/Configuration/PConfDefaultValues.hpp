//
//  PConfDefaultValues.h
//  BiSUNAOpenCL
//
//  Created by RHVT on 19/7/19.
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

#ifndef PConfDefaultValues_h
#define PConfDefaultValues_h

struct PCDefaultValues
{
    struct PCDVGeneral
    {
        static constexpr const bool binary = false;
        static constexpr const ushortT outIndex = 65535;
        static constexpr const char *exeType = "Thread";
        static constexpr const ushortT generations = 30;
    };

    struct PCDVPopulation
    {
        static constexpr const ushortT initialMutations = 200;
        static constexpr const ushortT stepMutations = 5;
        static constexpr const ushortT popSize = 100;
        static constexpr const ushortT noveltyMapSize = 20;
        // Add - Delete Neuron; Add - Delete Connection
        static constexpr const float addNeuron = 0.01;
        static constexpr const float delNeuron = 0.01;
        static constexpr const float addConn = 0.49;
        static constexpr const float delConn = 0.49;
        static constexpr const float neuroModProb = 0.1;
        static constexpr const float controlProb = 0.2;
        static constexpr const float weightMutProb = 0.5;
    };

    struct PCDVEnvironment
    {
        static constexpr const ushortT episodesPerAgent = 1;
        static constexpr const char *bisunaFile = "bisuna.dat";
        static constexpr const bool loadFromFile = false;
        static constexpr const bool saveToFile = false;
        static constexpr const bool saveEveryNGens = 10;
        static constexpr const char *envName = "MountainCar";
        static constexpr const char *envConf = "MountainCar.ini";
    };

    struct PCDVOpenCL
    {
        static constexpr const char *kernelFolder = "OCL/Kernels/";
        static constexpr const char *kernelStateName = "processStateG";
        static constexpr const bool kernelStateUseLocalVars = false;
        static constexpr const char *kernelEndEpisodeName = ""; //"endCellEpisode";
        static constexpr const char *kernelEvolveName = ""; // "spectrumDiversityEvolve";
        static constexpr const char *deviceType = "FPGA";
        static constexpr const char *oclFiles = "BiGlobal.xclbin";
        static constexpr const bool oclProfiling = false;
        static constexpr const long computeUnits = 1;
    };

    struct PCDVThread
    {
        static constexpr const ushortT threadNumber = 5;
    };

    struct PCDVContinuous
    {
        static constexpr const float maxWeight = 2147483647.0;
        static constexpr const float weightMutation = 1.0;
        static constexpr const float excitationThreshold = 0.0;
        static constexpr const float remainingNeuronThreshold = 0.001;
    };

    struct PCDVBinary
    {
        static constexpr const ushortT maxWeight = 65535;
        static constexpr const ushortT weightMutation = 1;
        static constexpr const ushortT excitationThreshold = 256;
        static constexpr const ushortT remainingNeuronThreshold = 15;
        static constexpr const ushortT excitationThresholdBits = 4;
        static constexpr const ushortT halfWeight = 255;
        static constexpr const ushortT midWeight = 32767;
    };

    struct PCDVDebugging
    {
        static constexpr const bool debugging = false;
        static constexpr const bool logging = false;
        static constexpr const ushortT seed = 1010;
    };
};

#endif /* PConfDefaultValues_h */
