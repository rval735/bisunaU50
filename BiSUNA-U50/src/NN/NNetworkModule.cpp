//
//  NNetworkModule.cpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 22/5/19.
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

#include "NNetworkModule.hpp"
#include "RandomUtils.hpp"
#include <cassert>
#include <algorithm>
#include <array>

// NNetworkModule constructor
NNetworkModule::NNetworkModule(const ushortT netID, const ushortT numInputs, const ushortT numOutputs)
{
    nParams = NNetworkParams(netID, numInputs, numOutputs);
    nNeurons = NNMFunction::interfaceNeurons(numInputs, numOutputs);
    nConns = vector<NConnection>();
}

// NNetworkModule destructor
NNetworkModule::~NNetworkModule()
{
}

///////////////////////////////////////////////////////////////
//    There are 3 types of structural mutation:
//        - topologic
//            -add/remove neuron
//            -add/remove connection
//            !!!!!!!Important: after removing a connection, a non-primer might become a primer and vice-versa!!!!
//            Threfore, always do a updatePrimerList() after structural Mutation
///////////////////////////////////////////////////////////////
void NNetworkModule::structuralMutation()
{
    NMutationType mt = NNMFunction::getMutationType();
    switch (mt) {
        case mtAddNeuron:
        {
            NNMFunction::addNeuron(this);
            break;
        }
        case mtRemoveNeuron:
        {
            ushortT inoutNeurons = nParams.nInputs + nParams.nOutputs;
            ushortT neuronsSize = nNeurons.size();
            ushortT removeNeuronIdx = RandomUtils::randomRangeUShort(inoutNeurons - 1, neuronsSize - 1);
            NNeuron n = NNMFunction::deleteNeuron(removeNeuronIdx, nNeurons);
            NNMFunction::deleteNeuronConnections(n.nID, nConns);
            break;
        }
        case mtAddConnection:
            NNMFunction::addConnection(nNeurons, &nConns);
            break;
        case mtRemoveConnection:
        {
            ushortT removeConnIdx = RandomUtils::randomPositive(nConns.size() - 1);
            NNMFunction::deleteConnection(removeConnIdx, &nConns);
            break;
        }
        default:
            // This case should never happen;
            printf("An error in function structuralMutation with an invalid case for MutationType");
            exit(1);
            break;
    }
}

// Creates a NNeuron vector that includes in its first places a number of inputs/outputs
// and makes sure that a vector contains at (numInputs + numOutputs)
vector<NNeuron> NNMFunction::interfaceNeurons(const ushortT numInputs, const ushortT numOutputs)
{
    ushortT nID = 0;
    ushortT minSize = numInputs + numOutputs;
    vector<NNeuron> neurons = vector<NNeuron>(minSize);
    
    for (ushortT i = 0; i < numInputs; ++i) {
        neurons[nID] = NNeuron(nID, frL1, ntInput);
        nID++;
    }
    
    for (ushortT i = 0; i < numOutputs; ++i) {
        neurons[nID] = NNeuron(nID, frL1, ntOutput);
        nID++;
    }
    
    return neurons;
}

// Using RandomUtils, it will return a possible mutation type value
// by the probabilities established in the parameter "MutationProbability"
// within the enum elements inside "NMutationType".
NMutationType NNMFunction::getMutationType()
{
    NMutationType mutationType = mtRemoveConnection;
    auto mutationChance = PConfig::globalProgramConfiguration()->mutationProb();
    float roulette = RandomUtils::randomPositiveFloat(1);
    float sum = 0;

    for (int i = 0; i < mtNumberMutationType; i++) {
        sum += mutationChance[i];
        if (roulette < sum) {
            mutationType = NMutationType(i);
            break;
        }
    }

    return mutationType;
}

// Create a random neuron with the specified neuronID
NNeuron NNMFunction::randomNeuron(const ushortT neuronID)
{
    NNeuron neuron = NNeuron();
    neuron.nID = neuronID;
    float controlConf = PConfig::globalProgramConfiguration()->neuroControlProb();
    
    // Consider the possibility this new element to become
    // a Control Neuron, given constant "CHANCE_OF_CONTROL_NEURON"
    if (RandomUtils::randomPositiveFloat(1) < controlConf) {
        //Create a Control Neuron
        neuron.nType = ntControl;
    }
    else {
        //Create a random neuron, which could include ntControl.
        neuron.nType = RandomUtils::randomNeuronType();
    }

    NFiringRate fr = RandomUtils::randomFiringRate();
    neuron.firingRate = fr;
    
    return neuron;
}

// Create a random connection that will have from and to references to neurons
// that fall between the range 0 to numNeurons.
NConnection NNMFunction::randomConnection(const vector<NNeuron> neurons)
{
    NConnection conn = NConnection();
    
    // Associate a random from/to nID to the new connection
    ushortT neuronsSize = neurons.size() - 1;
    ushortT randomFrom = RandomUtils::randomPositive(neuronsSize);
    ushortT randomTo = RandomUtils::randomPositive(neuronsSize);
    conn.fromNID = neurons[randomFrom].nID;
    conn.toNID = neurons[randomTo].nID;
    float neuroModConf = PConfig::globalProgramConfiguration()->neuroModulationProb();
    
    if (RandomUtils::randomPositiveFloat(1) < neuroModConf) {
        // This case, it will add an additional nID to the neuron that will
        // be neuromodulated
        ushortT randomMod = RandomUtils::randomPositive(neuronsSize);
        conn.neuroMod = neurons[randomMod].nID;
        conn.weight = 1;
    }
    else {
        conn.neuroMod = OUT_INDEX;
        int randomSign = RandomUtils::randomPositive(1);
        if (randomSign != 0) {
            conn.weight = -1;
        } else {
            conn.weight = 1;
        }
    }
    
    return conn;
}

/////////////////////////////
// MUTATIONS
////////////////////////////

inline float perturbationContinuous(const float &minMaxWeight, const float &weightPercentage, const float &cw)
{
    float variance = weightPercentage * cw;
    float perturbation = RandomUtils::randomRangeFloat(-variance, variance);
    float weight = cw + perturbation;
    
    // Verify that weights are within the boundaries of possible values
    weight = min(minMaxWeight, weight);
    weight = max(-minMaxWeight, weight);
    
    return weight;
}

inline ushortT pertubationBinary(const ushortT &weightPercentage, const ushortT &cw)
{
    ushortT varianceP = cw << weightPercentage;
    ushortT varianceN = cw >> weightPercentage;
    ushortT perturbation = RandomUtils::randomRangeUShort(varianceN, varianceP);
    // TODO!! Find the best perturbation operator for this case
    //    ParameterType weight = perturbation & cw;
    ushortT weight = perturbation | cw;
    return weight;
}

// Weight mutation modifies the weight value of a connection according to a specified variance,
// where neuromodulation connections are ignored.
void NNMFunction::weightMutation(NConnection *conn)
{
    // If the connection neuromodulates, it will not be mutated
    if (conn->neuroMod != OUT_INDEX) {
        return;
    }
    
    // In case that the random result does exceed the change of mutation, then the
    // connection will not be mutated
    double roulette = RandomUtils::randomPositiveFloat(1);
    float weightMutConf = PConfig::globalProgramConfiguration()->weightMutationProb();
    
    if (roulette > weightMutConf) {
        return;
    }
    
    ParameterType cw = conn->weight;
#ifdef CONTINUOUS_PARAM
    conn->weight = perturbationContinuous(MAXIMUM_WEIGHT, WEIGHT_MUTATION_CHANGE_PERCENTAGE, cw);
#else
    conn->weight = pertubationBinary(WEIGHT_MUTATION_CHANGE_PERCENTAGE, cw);
#endif
}

// This function creates a neuron, two connections and adds them to its passing
// reference containers. In case the neuron is control, it is added to the list
// of primers.
void NNMFunction::addNeuron(NNetworkModule *netModule)
{
    vector<NNeuron> &neurons = netModule->nNeurons;
    vector<NConnection> &conns = netModule->nConns;
    
    ushortT nID = neurons.size();
    
    // if it fails here, clNeuron size has exceeded the number of elements addressable by ushort
    assert(nID < OUT_INDEX);
    
    NNeuron neuron = randomNeuron(nID);
    NConnection conn0 = randomConnection(neurons);
    conn0.fromNID = neuron.nID;
    NConnection conn1 = randomConnection(neurons);
    conn1.toNID = neuron.nID;
    
    neurons.push_back(neuron);
    conns.push_back(conn0);
    conns.push_back(conn1);
}

// This function removes a "nID" chosen neuron from the vector "
NNeuron NNMFunction::deleteNeuron(const ushortT nID, vector<NNeuron> &neurons)
{
    NNeuron neuron = NNeuron();
    size_t neuronsSize = neurons.size();
    
    // This case should not happen, in case it does, it will return immidiatelly
    if (nID == OUT_INDEX || nID > neuronsSize) {
        return neuron;
    }
    
    if (neuronsSize <= 0) {
        printf("An error in function deleteNeuron with an invalid neurons vector");
        return neuron;
    }

    neuron = neurons[nID];
    NNeuronType removeType = neuron.nType;
    
    // If the neuron to be removed is an input/output, it will not make
    // any changes to the function
    if (removeType == ntInput || removeType == ntOutput) {
        return neuron;
    }

    auto deletedPos = neurons.erase(neurons.begin() + nID);
    
    // In order to keep the correct correspondance of nID with the position
    // in the vector, all values after the deleted neuron must reduce their
    // index by 1.
    for_each(deletedPos, neurons.end(), [](NNeuron &neuron){
        neuron.nID -= 1;
    });
    
    return neuron;
}

// This function removes all connections from/to/neuroMod referenced at "neuron"
// using its nID from a "conns" vector
void NNMFunction::deleteNeuronConnections(const ushortT nID, vector<NConnection> &conns)
{
    //remove all connections to this or from this neuron
    for (auto i = conns.begin(); i != conns.end(); i++) {
        if ((*i).fromNID == nID || (*i).toNID == nID || (*i).neuroMod == nID) {
            // Notice that the iterator is decremented after it is passed
            // to erase() but before erase() is executed
            // https://www.techiedelight.com/remove-elements-vector-inside-loop-cpp/
            conns.erase(i--);
            // An alternative: https://www.haroldserrano.com/blog/c-tip-17-how-to-remove-vector-elements-in-a-loop
        } else {
            if ((*i).fromNID > nID) {
                (*i).fromNID -= 1;
            }
            
            if ((*i).toNID > nID) {
                (*i).toNID -= 1;
            }
            
            ushortT neuroMod = (*i).neuroMod;
            if (neuroMod != OUT_INDEX && neuroMod > nID) {
                (*i).neuroMod -= 1;
            }
        }
    }
}

void NNMFunction::addConnection(const vector<NNeuron> neurons, vector<NConnection> *conns)
{
    NConnection conn = randomConnection(neurons);
    conns->push_back(conn);
}

void NNMFunction::deleteConnection(const ushortT connPos, vector<NConnection> *conns)
{
    size_t conSize = conns->size();
    if (conSize == 0) {
        return;
    }
    
    if (connPos > conSize || connPos < 0) {
        return;
    }
    
    conns->erase(conns->begin() + connPos);
}

////////////////////////////////////
// Neuron Filtering
////////////////////////////////////

// This function will check every ntControl/ntCPrimer neuron to verify if there are any
// connections to them from any ntControl/ntCPrimer, if there are not, then, it becomes a ntCPrimer.
void NNMFunction::updatePrimers(const vector<NConnection> &conns, vector<NNeuron> *neurons)
{
    size_t connsSize = conns.size();
    
    // For every neuron in the vector
    for (NNeuron &neuron : *neurons) {
        // Check if it is primer or control
        bool isControlOrPrimer = neuron.nType & (ntControl | ntCPrimer);
        if (isControlOrPrimer == false) {
            // No control or primer, continue execution
            continue;
        }
        
        bool isPrimer = true;
        // Then go over all connections and check which ones points to this neuron
        for (size_t i = 0; i < connsSize && isPrimer == true; i++) {
            NConnection conn = conns[i];
            bool toNID = conn.toNID != neuron.nID;
            bool recurrent = conn.fromNID == neuron.nID;
            if (toNID || recurrent) {
                continue;
            }
            
            // Primer neurons could potentially take input from any other type
            // of neurons but not from control or primers, that is what this
            // conditional checks.
            bool incomingNotCtrPrimer = !((*neurons)[conn.fromNID].nType & (ntControl | ntCPrimer));
            isPrimer = isPrimer && incomingNotCtrPrimer;
        }
        
        // It is possible that some neurons were ntPrimer and then after a mutation
        // received a connection, then they are no longer a primer. On the other hand
        // it is possible that the inverse happened, then it must be checked both cases.
        neuron.nType = isPrimer ? ntCPrimer : ntControl;
    }
}
