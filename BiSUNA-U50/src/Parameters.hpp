//
//  Parameters.h
//  BiSUNAOpenCL
//
//  Created by R on 15/5/19.
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

#ifndef PARAMETERS
#define PARAMETERS

#include <vector>
using namespace std;

// -----------  Unified Neural Model ------------- //

// Comment line below to make SUNA values discrete (binary).
// When it is continuous, it will behave as expected from the
// original publication, whereas commented, it will evolve
// binary neural networks only
//#define CONTINUOUS_PARAM
#define USING_OPENCL

#define DMA_ALIGNMENT 64

#ifdef USING_OPENCL
    #if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/opencl.h>
    #else
    #include <CL/cl.h>
    #endif

    using floatT = cl_float;
    using ushortT = cl_ushort;
    using boolT = cl_bool;
    using uintT = cl_uint;
    using ulongT = cl_ulong;
    using ulong = cl_ulong;
#else
    using floatT = float;
    using ushortT = unsigned short;
    using boolT = bool;
    using uintT = unsigned long int; // 32 bits
    using ulongT = unsigned long long int; // 64 bits
    using ulong = unsigned long long int; // 64 bits
#endif


#ifdef CONTINUOUS_PARAM
    using ParameterType = floatT;
    #define MAXIMUM_WEIGHT 2147483647.0
    #define WEIGHT_MUTATION_CHANGE_PERCENTAGE 1.0 //10 = 1000% change, 1 = 100% change possible
    #define EXCITATION_THRESHOLD 0.0    //minimum excitation necessary to activate the neuron
    #define EXCITATION_THRESHOLD_BITS 4 // Used only in the binary part, but redefined here for compilation purposes
    #define REMAINING_NEURON_THRESHOLD 0.001
    #define HALF_WEIGHT 0
    #define MID_WEIGHT 0
#else
    using ParameterType = ushortT;
    #define MAXIMUM_WEIGHT 65535
    #define WEIGHT_MUTATION_CHANGE_PERCENTAGE 1
    #define EXCITATION_THRESHOLD 256    //minimum excitation necessary to deactivate the neuron
// This constant considers the number of bits an excitation neuron must have in order
// to trigger its actions.
    #define EXCITATION_THRESHOLD_BITS 4
//    #define EXCITATION_THRESHOLD_BITS 6
// This represents the half part of all bits in a ushort (16 bits) set to one, it is used by primer, control
// and threshold neurons to check its trigger action
    #define HALF_WEIGHT 255
// Represents the actual mid section of the ushort type, use only in the discretization functions to transform
// between continuous inputs/outputs
    #define MID_WEIGHT 32767
    #define REMAINING_NEURON_THRESHOLD 15
#endif

using ParameterVector = vector<ParameterType>;

// This constant helps to keep track of the ushortT indexing values if for some
// rason another type is used, it should be changed accordingly, for example, if
// it changes to "int", OUT_INDEX should be considered to be "-1"
#define OUT_INDEX 65535

// Used as the initial value of an agent when executing in a new environment
#define EXTREME_NEGATIVE_REWARD -1000000 // ------------

#endif
