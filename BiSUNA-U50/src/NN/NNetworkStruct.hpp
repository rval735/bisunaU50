//
//  CLNetworkKernel.h
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

#ifndef NNetworkStruct_hpp
#define NNetworkStruct_hpp

#include "Configuration/PConfig.hpp"

enum NNeuronType: ushortT
{
    // The numbers here represent bits that can be activated
    // to pass along multiple types between ntID and ntControl
    // For example, ntID | ntThreshold would check for both types
    ntID = 16,
    ntActivation = 32,
    ntThreshold = 64,
    ntRandom = 128,
    ntControl = 256,
    ntRoll = 512,
    ntNumberNeuronType = 6,
    ///////////////////////////////////////////
    // ntInput and ntOutput are not considered
    // within the number of neuron type because
    // all network modules will have them
    ///////////////////////////////////////////
    ntCPrimer = 1024, // This is an special type of control that always activates, with
    // the principal characteristic of not receiving any connections to itself, when it does,
    // it is transformed into a simple ntControl neuron. Every structural mutation must
    // check if any of that happened in order to update this flag. It does not count as
    // a typical neuron type, for that reason is not before ntNumberNeuronType.
    ntInput = 1,
    ntOutput = 2
};

const vector<NNeuronType> ListedNNeuronType = {
    ntID, ntActivation, ntThreshold, ntRandom,
    ntControl, ntRoll, ntCPrimer, ntInput, ntOutput
};

enum NConnType: ushortT
{
    ctRecurrent,
    ctFeedForward,
    ctNumberConnType
};

// Predefined levels of firing rate
enum NFiringRate: ushortT
{
    frL1 = 1,
    frL7 = 7,
    frL49 = 49,
    frNumberFiringRate = 3
};

enum NMutationType: ushortT
{
    mtAddNeuron,
    mtRemoveNeuron,
    mtAddConnection,
    mtRemoveConnection,
    mtNumberMutationType
};

//------------------------------------------------
//------------------------------------------------
//------------------------------------------------

struct NNeuron
{
    ushortT nID;
    NFiringRate firingRate;
    NNeuronType nType;
    
    NNeuron(const ushortT neuronID = OUT_INDEX,
             const NFiringRate fr = frL1,
             const NNeuronType nt = ntID):
        nID(neuronID), firingRate(fr), nType(nt) {};
};

struct NConnection
{
    ushortT fromNID;
    ushortT toNID;
    ushortT neuroMod;    //-1 for inactive, for >0 it is active and represents the id of the neuron whose response is used as weight
    ParameterType weight;
    
    // A new connection will not be pointed by or to a neuron (fromID, toID) or have a valud neuro modulated index
    NConnection(const ushortT from = OUT_INDEX,
                 const ushortT to = OUT_INDEX,
                 const ushortT mod = OUT_INDEX,
                 const ParameterType cWeight = 0):
    fromNID(from), toNID(to), neuroMod(mod), weight(cWeight) {};
};

struct NNeuronState
{
    ParameterType state;
    ParameterType prevState;
    ParameterType excitation;
    boolT isFired;
    
    NNeuronState(): state(0), prevState(0), excitation(0), isFired(false) { };
};

struct NConnState
{
    ushortT connID;
    NConnType connType;
    NConnType prevConnType;
    
    NConnState(const ushortT cID = OUT_INDEX,
                const NConnType ct = ctRecurrent,
                const NConnType prev = ctRecurrent):
    connID(cID), connType(ct), prevConnType(prev) {};
};

struct NNetworkParams
{
    ushortT networkID; // Corresponding network ID
    ushortT nInputs; // How many neurons are inputs;
    ushortT nOutputs; // How many neurons are outputs;
    
    NNetworkParams(const ushortT netID = 0,
                    const ushortT inputs = 0,
                    const ushortT outputs = 0):
        networkID(netID), nInputs(inputs), nOutputs(outputs) {};
};

#endif
