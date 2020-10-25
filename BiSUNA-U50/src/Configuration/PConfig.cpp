//
//  PConfig.cpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 18/7/19.
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

#include "PConfig.hpp"
#include "PConfDefaultValues.hpp"
#include <iostream>
#include <sstream>
#include <string.h>
#include <array>

pair<bool, ushortT> findElem(const char *toFind, const ushortT elemsSize, const char *elems[]);
vector<string> split(const string &s, char delimiter);

PConfig::PConfig(const char *fileName):
    ini(fileName)
{
    if (ini.ParseError() != 0) {
        std::cout << "File: "<< fileName << " could not be loaded, finishing execution\n";
        exit(1);
    }
}

PConfig::~PConfig() {}

enum PCSectionVal: ushortT {
    PCSGeneral,
    PCSPopulation,
    PCSEnvironment,
    PCSOpenCL,
    PCSThread,
    PCSContinuous,
    PCSBinary,
    PCSDebugging,
};

const char *PCSectionStr[] =
{
    "General", "Population", "Environment",
    "OpenCL", "Thread", "Continuous",
    "Binary", "Debugging"
};


//-------------- General ------------

enum PCGeneralVal: ushortT
{
    PCGBinary,
    PCGOutIndex,
    PCGExecutionType,
    PCGGenerations
};

const char *PCGeneralStr[] = {"Binary", "OutIndex", "ExecutionType", "Generations"};
const char *PCGExecutionStr[] = {"Thread", "OpenCL", "OpenCLFull"};

bool PConfig::binaryNeurons()
{
    auto defaultValue = PCDefaultValues::PCDVGeneral::binary;
    return ini.GetBoolean(PCSectionStr[PCSGeneral], PCGeneralStr[PCGBinary], defaultValue);
}

ushortT PConfig::outIndex()
{
    auto defaultValue = PCDefaultValues::PCDVGeneral::outIndex;
    return ini.GetInteger(PCSectionStr[PCSGeneral], PCGeneralStr[PCGOutIndex], defaultValue);
}

PCGExecutionVal PConfig::exeType()
{
    auto defaultValue = PCDefaultValues::PCDVGeneral::exeType;
    auto res = ini.Get(PCSectionStr[PCSGeneral], PCGeneralStr[PCGExecutionType], defaultValue);
    auto isPos = findElem(res.c_str(), 2, PCGExecutionStr);
    if (isPos.first == false) {
        printf("Execution type not supported (%s). Terminating program", res.c_str());
        exit(EXIT_FAILURE);
    }

    return PCGExecutionVal(isPos.second);
}

uintT PConfig::generations()
{
    auto defaultValue = PCDefaultValues::PCDVGeneral::generations;
    return (uintT)ini.GetInteger(PCSectionStr[PCSGeneral], PCGeneralStr[PCGGenerations], defaultValue);
}

//-------------- Population ------------

enum PCPopulationVal: ushortT
{
    PCPNumberInitialMutations, PCPStepMutations, PCPPopulationSize,
    PCPNoveltyMapSize, PCPChanceAddNeuron, PCPChanceDelNeuron,
    PCPChanceAddConnection, PCPChanceDelConnection, PCPChanceNeuromodulation,
    PCPChanceControlNeuron, PCPChanceWeightMutation
};

const char *PCPopulationStr[] = {"NumberInitialMutations", "StepMutations", "PopulationSize",
    "NoveltyMapSize", "ChanceAddNeuron", "ChanceDelNeuron", "ChanceAddConnection",
    "ChanceDelConnection", "ChanceNeuromodulation", "ChanceControlNeuron", "ChanceWeightMutation"
};

ushortT PConfig::initialMutations()
{
    auto defaultValue = PCDefaultValues::PCDVPopulation::initialMutations;
    return ini.GetInteger(PCSectionStr[PCSPopulation], PCPopulationStr[PCPNumberInitialMutations], defaultValue);
}

ushortT PConfig::stepMutations()
{
    auto defaultValue = PCDefaultValues::PCDVPopulation::stepMutations;
    return ini.GetInteger(PCSectionStr[PCSPopulation], PCPopulationStr[PCPStepMutations], defaultValue);
}

ushortT PConfig::populationSize()
{
    auto defaultValue = PCDefaultValues::PCDVPopulation::popSize;
    return ini.GetInteger(PCSectionStr[PCSPopulation], PCPopulationStr[PCPPopulationSize], defaultValue);
}

ushortT PConfig::noveltyMapSize()
{
    auto defaultValue = PCDefaultValues::PCDVPopulation::noveltyMapSize;
    return ini.GetInteger(PCSectionStr[PCSPopulation], PCPopulationStr[PCPNoveltyMapSize], defaultValue);
}

array<float, 4> PConfig::mutationProb()
{
    float base = PCDefaultValues::PCDVPopulation::addNeuron;
    float addN = ini.GetReal(PCSectionStr[PCSPopulation], PCPopulationStr[PCPChanceAddNeuron], base);
    base = PCDefaultValues::PCDVPopulation::delNeuron;
    float delN = ini.GetReal(PCSectionStr[PCSPopulation], PCPopulationStr[PCPChanceDelNeuron], base);
    base = PCDefaultValues::PCDVPopulation::addConn;
    float addC = ini.GetReal(PCSectionStr[PCSPopulation], PCPopulationStr[PCPChanceAddConnection], base);
    base = PCDefaultValues::PCDVPopulation::delConn;
    float delC = ini.GetReal(PCSectionStr[PCSPopulation], PCPopulationStr[PCPChanceDelConnection], base);

    return {addN, delN, addC, delC};
}

float PConfig::neuroModulationProb()
{
    auto defaultValue = PCDefaultValues::PCDVPopulation::neuroModProb;
    return ini.GetReal(PCSectionStr[PCSPopulation], PCPopulationStr[PCPChanceNeuromodulation], defaultValue);
}

float PConfig::neuroControlProb()
{
    auto defaultValue = PCDefaultValues::PCDVPopulation::controlProb;
    return ini.GetReal(PCSectionStr[PCSPopulation], PCPopulationStr[PCPChanceControlNeuron], defaultValue);
}

float PConfig::weightMutationProb()
{
    auto defaultValue = PCDefaultValues::PCDVPopulation::weightMutProb;
    return ini.GetReal(PCSectionStr[PCSPopulation], PCPopulationStr[PCPChanceWeightMutation], defaultValue);
}


//-------------- Environment ------------

enum PCEnvironmentVal: ushortT
{
    PCEEpisodesPerAgent,
    PCEBiSUNAFile,
    PCELoadFromFile,
    PCESaveToFile,
    PCESaveEveryNGenerations,
    PCEEnvironmentName,
    PCEEnvironmentConf
};

const char *PCEnvironmentStr[] = {"EpisodesPerAgent", "BiSUNAFile", "LoadFromFile",
    "SaveToFile", "SaveEveryNGenerations", "EnvironmentName", "EnvironmentConf"
};

const char *PCEnvironmentNames[] = {"MountainCar", "DoubleCartPole", "FunctionApproximation",
    "GymEnv", "Multiplexer", "SingleCartPole", "SymmetricEncryption", "SymmetricEncryptionCPA",
    "RandomWalk"
};

ushortT PConfig::episodesPerAgent()
{
    auto defaultValue = PCDefaultValues::PCDVEnvironment::episodesPerAgent;
    return ini.GetInteger(PCSectionStr[PCSEnvironment], PCEnvironmentStr[PCEEpisodesPerAgent], defaultValue);
}

string PConfig::bisunaFile()
{
    auto defaultValue = PCDefaultValues::PCDVEnvironment::bisunaFile;
    return ini.Get(PCSectionStr[PCSEnvironment], PCEnvironmentStr[PCEBiSUNAFile], defaultValue);
}

bool PConfig::loadFromFile()
{
    auto defaultValue = PCDefaultValues::PCDVEnvironment::loadFromFile;
    return ini.GetBoolean(PCSectionStr[PCSEnvironment], PCEnvironmentStr[PCELoadFromFile], defaultValue);
}

bool PConfig::saveToFile()
{
    auto defaultValue = PCDefaultValues::PCDVEnvironment::saveToFile;
    return ini.GetBoolean(PCSectionStr[PCSEnvironment], PCEnvironmentStr[PCESaveToFile], defaultValue);
}

ushortT PConfig::saveEveryNGenerations()
{
    auto defaultValue = PCDefaultValues::PCDVEnvironment::saveEveryNGens;
    return ini.GetInteger(PCSectionStr[PCSEnvironment], PCEnvironmentStr[PCESaveEveryNGenerations], defaultValue);
}

PCEnvironmentSupported PConfig::environmentName()
{
    string val = ini.Get(PCSectionStr[PCSEnvironment], PCEnvironmentStr[PCEEnvironmentName], "");
    auto isPos = findElem(val.c_str(), PCEnvironmentSupportedNumber, PCEnvironmentNames);

    if (isPos.first == false) {
        printf("This environment (%s) is not currently supported, execution will terminate", val.c_str());
        exit(EXIT_FAILURE);
    }

    return PCEnvironmentSupported(isPos.second);
}

string PConfig::environmentConf()
{
    return ini.Get(PCSectionStr[PCSEnvironment], PCEnvironmentStr[PCEEnvironmentConf], "");
}


//-------------- OpenCL ------------

enum PCOpenCLVal: ushortT
{
    PCOKernelFolder,
    PCOKernelStateName,
    PCOKernelStateUseLocalVars,
    PCOKernelEndEpisodeName,
    PCOKernelEvolveName,
    PCODeviceType,
    PCOOCLFiles,
    PCOOCLProfiling,
    PCOComputeUnits
};

const char *PCOpenCLStr[] = {"KernelFolder", "KernelStateName", "KernelStateUseLocalVars", "KernelEndEpisodeName", "KernelEvolveName",
                             "DeviceType", "OCLFiles", "OCLProfiling", "ComputeUnits"};

string PConfig::kernelFolder()
{
    auto defaultValue = PCDefaultValues::PCDVOpenCL::kernelFolder;
    return ini.Get(PCSectionStr[PCSOpenCL], PCOpenCLStr[PCOKernelFolder], defaultValue);
}

string PConfig::kernelStateName()
{
    auto defaultValue = PCDefaultValues::PCDVOpenCL::kernelStateName;
    return ini.Get(PCSectionStr[PCSOpenCL], PCOpenCLStr[PCOKernelStateName], defaultValue);
}

bool PConfig::kernelStateUseLocalVars()
{
    auto defaultValue = PCDefaultValues::PCDVOpenCL::kernelStateUseLocalVars;
    return ini.GetBoolean(PCSectionStr[PCSOpenCL], PCOpenCLStr[PCOKernelStateUseLocalVars], defaultValue);
}

string PConfig::kernelEndEpisodeName()
{
    auto defaultValue = ""; // PCDefaultValues::PCDVOpenCL::kernelEndEpisodeName;
    return ini.Get(PCSectionStr[PCSOpenCL], PCOpenCLStr[PCOKernelEndEpisodeName], defaultValue);
}

string PConfig::kernelEvolveName()
{
    auto defaultValue = ""; // PCDefaultValues::PCDVOpenCL::kernelEvolveName;
    return ini.Get(PCSectionStr[PCSOpenCL], PCOpenCLStr[PCOKernelEvolveName], defaultValue);
}

uintT PConfig::deviceType()
{
    auto defaultValue = PCDefaultValues::PCDVOpenCL::deviceType;
    string dt = ini.Get(PCSectionStr[PCSOpenCL], PCOpenCLStr[PCODeviceType], defaultValue);

    if (dt.compare("CPU") == 0) {
        return 2;
    }

    if (dt.compare("GPU") == 0) {
        return 4;
    }

    if (dt.compare("FPGA") == 0) {
        return 8;
    }

    return 0;
}

vector<string> PConfig::oclFiles()
{
    auto defaultValue = PCDefaultValues::PCDVOpenCL::oclFiles;
    const string dt = ini.Get(PCSectionStr[PCSOpenCL], PCOpenCLStr[PCOOCLFiles], defaultValue);
    return split(dt, ',');
}

bool PConfig::oclProfiling()
{
    auto defaultValue = PCDefaultValues::PCDVOpenCL::oclProfiling;
    return ini.GetBoolean(PCSectionStr[PCSOpenCL], PCOpenCLStr[PCOOCLProfiling], defaultValue);
}

long PConfig::computeUnits()
{
    auto defaultValue = PCDefaultValues::PCDVOpenCL::computeUnits;
    return ini.GetInteger(PCSectionStr[PCSOpenCL], PCOpenCLStr[PCOComputeUnits], defaultValue);
}

//-------------- Thread ------------

enum PCThreadVal: ushortT
{
    PCTThreadNumber
};

const char *PCThreadStr[] = {"ThreadNumber"};

ushortT PConfig::threadNumber()
{
    auto defaultValue = PCDefaultValues::PCDVThread::threadNumber;
    return ini.GetInteger(PCSectionStr[PCSThread], PCThreadStr[PCTThreadNumber], defaultValue);
}


//-------------- Continuous  ------------

enum PCContinuousVal: ushortT
{
    PCCMaxWeight,
    PCCWeightMutationChangePercentage,
    PCCExcitationThreshold,
    PCCRemainingNeuronThreshold
};

const char *PCContinuousStr[] = {"MaxWeight", "WeightMutationChangePercentage", "ExcitationThreshold", "RemainingNeuronThreshold"};

float PConfig::maxWeightC()
{
    auto defaultValue = PCDefaultValues::PCDVContinuous::maxWeight;
    return ini.GetReal(PCSectionStr[PCSContinuous], PCContinuousStr[PCCMaxWeight], defaultValue);
}

float PConfig::weightMutationChangeC()
{
    auto defaultValue = PCDefaultValues::PCDVContinuous::weightMutation;
    return ini.GetReal(PCSectionStr[PCSContinuous], PCContinuousStr[PCCWeightMutationChangePercentage], defaultValue);
}
float PConfig::excitationThresholdC()
{
    auto defaultValue = PCDefaultValues::PCDVContinuous::excitationThreshold;
    return ini.GetReal(PCSectionStr[PCSContinuous], PCContinuousStr[PCCExcitationThreshold], defaultValue);
}

float PConfig::remainingNeuronThresholdC()
{
    auto defaultValue = PCDefaultValues::PCDVContinuous::excitationThreshold;
    return ini.GetReal(PCSectionStr[PCSContinuous], PCContinuousStr[PCCRemainingNeuronThreshold], defaultValue);
}


//-------------- Binary ------------

enum PCBinaryVal: ushortT
{
    PCBMaxWeight,
    PCBWeightMutationChangePercentage,
    PCBExcitationThreshold,
    PCBRemainingNeuronThreshold,
    PCBExcitationThresholdBits,
    PCBHalfWeight,
    PCBMidWeight
};

const char *PCBinaryStr[] = {"MaxWeight", "WeightMutationChangePercentage", "ExcitationThreshold", "RemainingNeuronThreshold",
    "ExcitationThresholdBits", "HalfWeight", "MidWeight"
};

ushortT PConfig::maxWeightB()
{
    auto defaultValue = PCDefaultValues::PCDVBinary::maxWeight;
    return ini.GetInteger(PCSectionStr[PCSBinary], PCBinaryStr[PCBMaxWeight], defaultValue);
}

ushortT PConfig::weightMutationChangeB()
{
    auto defaultValue = PCDefaultValues::PCDVBinary::weightMutation;
    return ini.GetInteger(PCSectionStr[PCSBinary], PCBinaryStr[PCBWeightMutationChangePercentage], defaultValue);
}

ushortT PConfig::excitationThresholdB()
{
    auto defaultValue = PCDefaultValues::PCDVBinary::excitationThreshold;
    return ini.GetInteger(PCSectionStr[PCSBinary], PCBinaryStr[PCBExcitationThreshold], defaultValue);
}

ushortT PConfig::remainingNeuronThresholdB()
{
    auto defaultValue = PCDefaultValues::PCDVBinary::remainingNeuronThreshold;
    return ini.GetInteger(PCSectionStr[PCSBinary], PCBinaryStr[PCBRemainingNeuronThreshold], defaultValue);
}

ushortT PConfig::excitationThresholdBitsB()
{
    auto defaultValue = PCDefaultValues::PCDVBinary::excitationThresholdBits;
    return ini.GetInteger(PCSectionStr[PCSBinary], PCBinaryStr[PCBExcitationThresholdBits], defaultValue);
}

ushortT PConfig::halfWeightB()
{
    auto defaultValue = PCDefaultValues::PCDVBinary::halfWeight;
    return ini.GetInteger(PCSectionStr[PCSBinary], PCBinaryStr[PCBHalfWeight], defaultValue);
}

ushortT PConfig::midWeightB()
{
    auto defaultValue = PCDefaultValues::PCDVBinary::midWeight;
    return ini.GetInteger(PCSectionStr[PCSBinary], PCBinaryStr[PCBMidWeight], defaultValue);
}


//-------------- Debugging ------------

enum PCDebuggingVal: ushortT
{
    PCDEnableDebug,
    PCDEnableLogging,
    PCDRandomSeed
};

const char *PCDebuggingStr[] = {"EnableDebug", "EnableLogging", "RandomSeed"};

bool PConfig::enableDebugging()
{
    auto defaultValue = PCDefaultValues::PCDVDebugging::debugging;
    return ini.GetBoolean(PCSectionStr[PCSDebugging], PCDebuggingStr[PCDEnableDebug], defaultValue);
}

bool PConfig::enableLogging()
{
    auto defaultValue = PCDefaultValues::PCDVDebugging::logging;
    return ini.GetBoolean(PCSectionStr[PCSDebugging], PCDebuggingStr[PCDEnableLogging], defaultValue);
}

ushortT PConfig::randomSeed()
{
    auto defaultValue = PCDefaultValues::PCDVDebugging::seed;
    return ini.GetInteger(PCSectionStr[PCSDebugging], PCDebuggingStr[PCDRandomSeed], defaultValue);
}

//--------------------------------------------
static PConfig *globalConf = NULL;
PConfig *PConfig::globalProgramConfiguration(const char *fileName)
{
    if (globalConf == NULL) {
        // TODO!! Check cases for "null" file name and not having a default
        // "BiSUNAConf.ini" file where the executable is.
        globalConf = new PConfig(fileName);
    }

    return globalConf;
}

void PConfig::discardConfiguration()
{
    if (globalConf != NULL) {
        delete globalConf;
        globalConf = NULL;
    }
}
//--------------------------------------------

pair<bool, ushortT> findElem(const char *toFind, const ushortT elemsSize, const char *elems[])
{
    ushortT pos = 0;
    for (; pos < elemsSize; pos++) {
        auto res = strcmp(toFind, elems[pos]);
        if (res == 0) {
            return {true, pos};
        }
    }

    return {false, pos};
}

// https://www.fluentcpp.com/2017/04/21/how-to-split-a-string-in-c/
vector<string> PConfig::split(const string &s, char delimiter)
{
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);

    while (getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }

    return tokens;
}
