//
//  CLNetworkStruct.h
//  BiSUNAOpenCL
//
//  Created by RHVT on 24/7/19.
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

#ifndef CLNetworkStruct_h
#define CLNetworkStruct_h

//#define CONTINUOUS_PARAM

#ifdef CONTINUOUS_PARAM
    typedef float CLPType;
    #ifndef MAXIMUM_WEIGHT
        #define MAXIMUM_WEIGHT 2147483647.0
    #endif
    #define EXCITATION_THRESHOLD 0.0    //minimum excitation necessary to activate the neuron
    #define REMAINING_NEURON_THRESHOLD 0.001

    #define WEIGHT_MUTATION_CHANGE_PERCENTAGE 1.0 //10 = 1000% change, 1 = 100% change possible
    #define EXCITATION_THRESHOLD_BITS 4 // Used only in the binary part, but redefined here for compilation purposes
    #define HALF_WEIGHT 0  // Used only in the binary part, but redefined here for compilation purposes
    #define MID_WEIGHT 0  // Used only in the binary part, but redefined here for compilation purposes
#else
    typedef ushort CLPType;
    #define EXCITATION_THRESHOLD 256//minimum excitation necessary to deactivate the neuron
    // This constant considers the number of bits an excitation neuron must have in order
    // to trigger its actions.
    #define REMAINING_NEURON_THRESHOLD 15
    #ifndef MAXIMUM_WEIGHT
        #define MAXIMUM_WEIGHT USHRT_MAX // 65535
    #endif
    #define WEIGHT_MUTATION_CHANGE_PERCENTAGE 1
    #define EXCITATION_THRESHOLD_BITS 4
//    #define EXCITATION_THRESHOLD_BITS 6
// This represents the half part of all bits in a ushort (16 bits) set to one, it is used by primer, control
// and threshold neurons to check its trigger action
    #define HALF_WEIGHT 255
// Represents the actual mid section of the ushort type, use only in the discretization functions to transform
// between continuous inputs/outputs
    #define MID_WEIGHT 32767
#endif

#ifndef INITIALBATCH
#define INITIALBATCH 1000
#endif

#ifndef INITIALBATCHX2
#define INITIALBATCHX2 (INITIALBATCH * 2)
#endif

#ifndef CELL_POPULATION
#define CELL_POPULATION 100
#endif

#ifndef NMSIZE
#define NMSIZE 20
#endif

#ifndef INPUT_SIZE
#define INPUT_SIZE 2
#endif

#ifndef OUTPUT_SIZE
#define OUTPUT_SIZE 3
#endif

#define ARRAY_MAX 128
#define MAX_FIRING_RATE 3
#define MAX_NEURON_TYPE 5

#ifndef OUT_INDEX
#define OUT_INDEX USHRT_MAX // 65535
#endif

// PConfig default values
#ifndef CHANCE_OF_CONTROL_NEURON
#define CHANCE_OF_CONTROL_NEURON 20 // 20% -> 0.2
#endif

#ifndef CHANCE_OF_NEUROMODULATION
#define CHANCE_OF_NEUROMODULATION 10 // 10% -> 0.1
#endif

#ifndef CHANCE_OF_ADD_NEURON
#define CHANCE_OF_ADD_NEURON 2 // 1% -> 0.01
#endif

#ifndef CHANCE_OF_DEL_NEURON
#define CHANCE_OF_DEL_NEURON 2 // 1% -> 0.01
#endif

#ifndef CHANCE_OF_ADD_CONN
#define CHANCE_OF_ADD_CONN 48 // 49% -> 0.49
#endif

#ifndef CHANCE_OF_DEL_CONN
#define CHANCE_OF_DEL_CONN 48 // 49% -> 0.49
#endif

#ifndef CHANCE_OF_WEIGHT_MUT
#define CHANCE_OF_WEIGHT_MUT 50  // weightMutProb 50% -> 0.5
#endif

#ifndef STEP_MUTATION
#define STEP_MUTATION 5
#endif

#ifndef EPISODES_PER_AGENT
#define EPISODES_PER_AGENT 1
#endif

typedef struct __attribute__((packed)) __attribute__((aligned(4)))
{
    CLPType clWeight;
    ushort clFromNID;
    ushort clToNID;
    ushort clNeuroMod;
    
    // Part of the CLConnState
    bool clCnType; // NOTE: false = ctRecurrent, true = ctForward
    bool clPCnType; // NOTE: false = ctRecurrent, true = ctForward
    // Part of the CLConnState
} CLConnection;


typedef struct __attribute__((packed)) __attribute__((aligned(4)))
{
    // Part of CLNeuronState
    CLPType clSt;
    CLPType clPSt;
    CLPType clExc;
    // Part of CLNeuronState
    
    // Part of CLNeuron
    ushort clNID;
    
    // clFiringRate: index 1 (6 most significant bits 1111111 0000000000)
    // clNType: index 1 (10 least significant bits 000000 1111111111)
    ushort clFRNT;
    // Part of CLNeuron
    
    bool clFired;
} CLNeuron;


typedef struct __attribute__((packed)) __attribute__((aligned(4)))
{
    CLConnection clCons[INITIALBATCHX2];
    CLNeuron clNrs[INITIALBATCH];
    float clFitness; // TODO!! check later on for a full BNN execution
    uint cellStep;
    ushort clNetworkID;
    ushort nrsSize;
    ushort connSize;
} CLCell;


typedef struct __attribute__((packed)) __attribute__((aligned(4)))
{
    ulong weight;
    ushort cellRef; // In order to keep values referenced in this cell
} CLNMStr;


typedef struct __attribute__((packed)) __attribute__((aligned(4)))
{
    ushort ciDist;
    ushort ciIndex;
} CLNMStrIndex;


typedef struct __attribute__((packed)) __attribute__((aligned(4)))
{
    CLNMStr nmStrs[NMSIZE];
    CLNMStrIndex lastIdx;
} CLNoveltyMap;


typedef struct __attribute__((packed)) __attribute__((aligned(4)))
{
    CLCell cells[CELL_POPULATION];
    CLNoveltyMap nmap;
    uint clGeneration;
    uint clSteps;
} CLUnifiedNeuralModel;

// Because in OCL it is messy to send pointers, instead the whole data dump is sent to the
// device, to then use offsets to access sections of that data. For that reason, this and
// CLNetworkModule have "ranges" to point to parts of data it corresponds. For example,
// an OCL device would have the pointer to the whole array of CLNeuronState, but this
// struct would allow to access only those corresponding to a particular clModID.
//    ushort nStStart; // Where NeuronState starts for this NetworkState
//    ushort nStEnd; // Where NeuronState ends for this NetworkState
//    ushort cStStart; // Where Connection state starts for this NetworkState
//    ushort cStEnd; // Where Connection state ends for this NetworkState
//    ushort inputSize;
//    ushort inputIdx; // The location assigned to this state that corresponds to Input values
//    ushort outputIdx; // The location assigned to this state that corresponds to Output values
//    // In order to assing the correct input/output value to a neuron, it should add its nID in
//    // case of inputs. With outputs, nID should reduce the "inputSize" in order to correctly
//    // assign the value. Example:
//    // OCL Input buffer: [1,2,3,4,5,6,7,8,9,10] // 10 elements shared for all modules
//    // OCL Output buffer: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] // 15 elements shared for all modules
//    // InputSize: 2; // Each network state has 2 input neurons
//    // inputIdx: 6 // This means this CLNetworkState its first input index is located in the 6th position
//    // OutputIdx: 10 // This CLNetworkState can write positions [10,11,12].
//    // If neuron output "2" wants to write, then it will locate its correct position as follows:
//    // ushort offset = outputIdx + (neuron.nID - inputSize);
//
//    ushort clModID; // The index of the Network module this state is related to
// CLNetworkState: nStStart, nStEnd, cStStart, cStEnd, inputSize, inputIdx, outputIdx, outputSize
// ushort8 netSt:  0         1        2          3       4           5        6           7

#endif /* CLNetworkStruct_h */
