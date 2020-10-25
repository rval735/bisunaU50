//
//  NNetworkState.cpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 27/5/19.
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

#include "NNetworkState.hpp"
#include <stack>
#include <list>
#include <numeric>
#include <cassert>
#include <algorithm>
#include "RandomUtils.hpp"

// NNetworkState constructors
NNetworkState::NNetworkState(NNetworkModule *ptrModule):
    module(ptrModule)
{
    // If it crashed here, it means a null pointer was passed as
    // ptrModule, which is not correct and should be resolved
    assert(module);
    
    size_t neuronsSize = module->nNeurons.size();
    size_t connsSize = module->nConns.size();
    
    nNSt = vector<NNeuronState>(neuronsSize);
    nCSt = vector<NConnState>(connsSize);
    // Surprised how iota function correctly sets the value of
    // connID to the sequence of numbers.
    iota(nCSt.begin(), nCSt.end(), 0);
    
    for (NNeuron n : module->nNeurons) {
        if (!(n.nType & (ntInput | ntOutput))) {
            continue;
        }
            
        ioVal[n.nID] = 0;
    }
}

// NNetworkState destructor
NNetworkState::~NNetworkState()
{
}

void NNSFunction::resetState(NNetworkState *st)
{
    for_each(st->nNSt.begin(), st->nNSt.end(), [](NNeuronState &nSt){
        nSt.excitation = 0;
        nSt.isFired = false;
        nSt.state = 0;
        nSt.prevState = 0;
    });
    
    for_each(st->nCSt.begin(), st->nCSt.end(), [](NConnState &cSt){
        cSt.connType = ctRecurrent;
        cSt.prevConnType = ctRecurrent;
    });
}

////////////////////////////////////////////////
////////////// Static Methods //////////////////
////////////////////////////////////////////////
/**
 Take a ParameterType vector, an Network state and perform operation "execute" on
 neurons type "ntInput". NOTE!! if the input has a different size compared to

 @param input Vector of elements that are represent the input to the neural network
 @param netSt Network state that is going to be modified
 */
void NNSFunction::processInputNeurons(const vector<ParameterType> &input, NNetworkState *netSt)
{
    const NNetworkParams &params = netSt->module->nParams;
    
    // Execute all Input Neurons that are excited (they are all not activated yet so no check is needed)
    // By design, all input neurons are located at the very beginning of the vector, therefore indexes
    // should match to the input provided.
    for (ushortT i = 0; i < params.nInputs; i++) {
        netSt->ioVal[i] = input[i];
        execute(i, netSt);
    }
    
    for (ushortT i = 0; i < params.nInputs; i++) {
        // Set neuron as already activated, aka isFired as true.
        netSt->nNSt[i].isFired = true;
    }
}

vector<ParameterType> NNSFunction::processOutputNeurons(NNetworkState *netSt)
{
    NNetworkParams params = netSt->module->nParams;
    ushortT firstOutput = params.nInputs;
    ushortT lastOutput = firstOutput + params.nOutputs;
    vector<ParameterType> output;
    output.reserve(lastOutput - firstOutput);
    
    for (int i = firstOutput; i < lastOutput; i++) {
        // In case output neurons have not fired yet, we need to
        // execute them here.
        if (netSt->nNSt[i].isFired == false) {
            execute(i, netSt);
            netSt->nNSt[i].isFired = true;
        }
        
        ParameterType outputVal = netSt->ioVal.at(i);
        output.push_back(outputVal);
    }
    
    return output;
}


void NNSFunction::processCPrimerNeurons(NNetworkState *netSt)
{
    const vector<NNeuron> &neurons = netSt->module->nNeurons;
    const vector<NConnection> &conns = netSt->module->nConns;
    vector<ushortT> primersRef;
    for (NNeuron neuron : neurons) {
        if (neuron.nType != ntCPrimer) {
            continue;
        }
        
        NNSFunction::execute(neuron.nID, netSt);
        primersRef.push_back(neuron.nID);
    }
    
    for (ushortT nID : primersRef) {
        netSt->nNSt[nID].isFired = true;
    }
    
    for (ushortT nID : primersRef) {
        NNSFunction::runControlID(nID, conns, &(netSt->nNSt));
    }
}

void NNSFunction::processControlNeurons(NNetworkState *netSt)
{
    const vector<NNeuron> &neurons = netSt->module->nNeurons;
    const vector<NConnection> &conns = netSt->module->nConns;
    
    //the loop only stops when:
    //    -no Control Neuron that is excited and was not activated is found
    for (;;) {
        stack<ushortT> activeNeurons;
        stack<ushortT> activeNeuronsCopy;
        
        //execute all Control Neurons that are excited and not activated
        for (auto i = neurons.begin(); i != neurons.end(); i++) {
            NNeuronType type = (*i).nType;
            ParameterType ne = netSt->nNSt[(*i).nID].excitation;
            bool isControl = type & (ntControl | ntCPrimer);
            bool isExcited = checkExcitationThreshold(ne);
            bool notFired = netSt->nNSt[(*i).nID].isFired == false;
            if (isControl && isExcited && notFired) {
                execute((*i).nID, netSt);
                activeNeurons.push((*i).nID);
            }
        }
        
        //stop loop if no excited and not activated control neuron was found
        if (activeNeurons.empty()) {
            break;
        }
        
        //set all Control Neurons as already activated and fire them
        while (!activeNeurons.empty()) {
            //int id= active_neuron;
            ushortT index = activeNeurons.top();
            
            //remove the top of the stack & add this in the copy of this stack
            activeNeurons.pop();
            activeNeuronsCopy.push(index);
            
            //mark as fired
            netSt->nNSt[index].isFired = true;
        }
        
        //Excite/Inhibit other neurons
        while (!activeNeuronsCopy.empty()) {
            ushortT controlID = activeNeuronsCopy.top();
            activeNeuronsCopy.pop();
            runControlID(controlID, conns, &(netSt->nNSt));
        }
    }
}

void NNSFunction::runControlID(const ushortT controlID, const vector<NConnection> &conns, vector<NNeuronState> *neuronsSt)
{    
    vector<NNeuronState> &replacer = *neuronsSt;
    
    // Check the connections from this neuron
    for (auto i = conns.begin(); i != conns.end(); i++) {
        
        // Skip if it is not a connection from this Neuron
        if ((*i).fromNID != controlID) {
            continue;
        }
        
        ushortT destination = (*i).toNID;
        ushortT modulator = (*i).neuroMod;
        
        ParameterType ns = replacer[controlID].state;
        ParameterType cw = (*i).weight;
        //no neuro modulation
        if (modulator == OUT_INDEX) {
            neuronOpr(cw, ns, replacer[destination].excitation);
        }
        //with neuro modulation
        else {
            ParameterType modulatorInput = 0;
            ParameterType neM = replacer[modulator].excitation;
            //check if it is excited
            if (checkExcitationThreshold(neM)) {
                //modulator already activated
                if (replacer[modulator].isFired == true) {
                    modulatorInput = replacer[modulator].state;
                }
                //modulator not already activated, use the state from the previous iteration
                else {
                    modulatorInput = replacer[modulator].prevState;
                }
            }
            
            neuronOpr(modulatorInput, ns, replacer[destination].excitation);
        }
    }
}

void NNSFunction::processRemainingNeurons(NNetworkState *netSt)
{
    list<ushortT> remainingNeuronsList;
    
    const vector<NNeuron> &neurons = netSt->module->nNeurons;
    
    //build the list of remaining neurons
    for (auto i = neurons.begin(); i != neurons.end(); i++) {
        ushortT nID = (*i).nID;
        NNeuronType type = (*i).nType;
        ParameterType ne = netSt->nNSt[nID].excitation;
        bool notControl = !(type & (ntControl | ntCPrimer));
        bool isExcited = checkExcitationThreshold(ne);
        bool notFired = netSt->nNSt[nID].isFired == false;
        
        if (notControl && isExcited && notFired) {
            remainingNeuronsList.push_back(nID);
        }
    }
        
    //the loop only stops when:
    //    -no Remaining Neuron, that was excited and not activated, is found
    for (;;) {
        stack<ushortT> activeNeurons;
        
        for (auto i = remainingNeuronsList.begin(); i != remainingNeuronsList.end(); i++) {
            ParameterType output = execute(*i, netSt, true);
            
            if (remainingNeuronThreshold(output)) {
                activeNeurons.push(*i);
            }
        }
        
        //stop loop if no excited and not activated control neuron was found
        if (activeNeurons.empty()) {
            break;
        }
        
        //set current active Neurons as already activated and fire them
        while (!activeNeurons.empty()) {
            ushortT nID = activeNeurons.top();
            
            //remove the top of the stack
            activeNeurons.pop();
            
            //mark as fired
            netSt->nNSt[nID].isFired = true;
            
            //remove activated neuron from the list of remaining neurons
            remainingNeuronsList.remove(nID);
        }
    }
}

//
// States: activated/deactivated, excited/inhibited, called/uncalled
//
// Sequence of activations:
//
//     example: [condition] type of neuron
//
//    **Control neurons has an activation bias of +1
//
// - Primers (Control Neurons (CN) that does not receive input from CN)
//
// - [excited and deactivated] CN
//
// - [excited] Input Neurons
//
// - [excited and deactivated and called] Neurons
//
vector<ParameterType> NNSFunction::process(const vector<ParameterType> &input, NNetworkState *netSt)
{
    ///////////////////////////// Input Neurons ////////////////////////////
    processInputNeurons(input, netSt);
    ///////////////////////////// CPrimer Neurons //////////////////////////
    
    processCPrimerNeurons(netSt);
    ///////////////////////////// Control Neurons //////////////////////////
    
    processControlNeurons(netSt);
    ///////////////////////////// Other Neurons ////////////////////////////
    
    processRemainingNeurons(netSt);
    ///////////////////////////// Output Neurons //////////////////////////
    
    vector<ParameterType> output = processOutputNeurons(netSt);
    ///////////////////////////// Cleaning Phase //////////////////////////
    
    //Clean Information:
    //
    //fire - fire clean the fire information
    //state - change neuron_state to previous neuron state
    //excitation - clear excitation
    for (auto i = netSt->nNSt.begin(); i != netSt->nNSt.end(); i++) {
        (*i).prevState = (*i).state;
        (*i).state = 0;
        (*i).excitation = 0;
        (*i).isFired = false;
    }
    
    for (auto i = netSt->nCSt.begin(); i != netSt->nCSt.end(); i++) {
        (*i).prevConnType = (*i).connType;
        (*i).connType = ctRecurrent;
    }
    
    return output;
}

// Basically, do:
//
//    w_i - weights
//    I_i - inputs
//
//    f( \sum w_i * I_i )
//
//    (Neuron0)---I0---w0--
//                     |
//    (Neuron1)---I1---w1---(This Neuron)-----
//                        |
//    (Neuron2)---I2---w2--
//
//    Ignore Input I_i if:
//        -Neuron_i is a control neuron
//
//    IMPORTANT: this function returns this neuron's firing signal
//           if the neuron is going to fire or not this signal is left undecided
//
ParameterType NNSFunction::execute(const ushortT &nID, NNetworkState *netSt, const bool &ignoreIfOnlyRecurrent)
{
    const vector<NNeuron> &neurons = netSt->module->nNeurons;
    const vector<NConnection> &conns = netSt->module->nConns;
    
    stack<NConnState> usedConns;
    ParameterType sum = 0;
    bool onlyRecurrentInputs = true;
    
    //check the connections to this neuron
    for (auto i = conns.begin(); i != conns.end(); i++) {
        
        //skip if it is not a connection to this Neuron
        if ((*i).toNID != nID) {
            continue;
        }
        
        ushortT source = (*i).fromNID;
        ushortT connPosition = distance(conns.begin(), i);

        //add this input multiplied by the weight of the connection
        //IGNORE if it comes from a Control Neuron
        NNeuronType t = neurons[source].nType;
        bool notControl = !(t & (ntControl | ntCPrimer));
        if (notControl) {
            
            ushortT modulator = (*i).neuroMod;
            //no neuro modulation
            if (modulator == OUT_INDEX) {
                ParameterType cw = (*i).weight;
                
                //check if they were already activated
                if (netSt->nNSt[source].isFired == true) {
                    ParameterType ns = netSt->nNSt[source].state;
                    neuronOpr(cw, ns, sum);

                    onlyRecurrentInputs = false;
                    
                    NConnState cnn = NConnState(connPosition, ctFeedForward);
                    
                    //marking the recurrent connection as used to be processed later on
                    //because we still do not know if this neuron will fire
                    usedConns.push(cnn);
                }
                //recurrent connection
                //case it was not activated yet, use the state from the previous iteration
                else {
                    ParameterType pn = netSt->nNSt[source].prevState;
                    
                    if (netSt->nCSt[connPosition].prevConnType != ctFeedForward) {
                        neuronOpr(cw, pn, sum);
                        NConnState cnn = NConnState(connPosition, ctRecurrent);
                        
                        //marking the recurrent connection as used to be processed later on
                        //because we still do not know if this neuron will fire
                        usedConns.push(cnn);
                    }
                }
            }
            //with neuron modulation
            else {
                ///////////////////////////////////////////////////////////////
                //
                //    Neuron Modulation
                //
                //
                //    ----(Modulator Neuron)----
                //                 |
                //    ----(Source Neuron)---------------(This Neuron)---------
                //
                ///////////////////////////////////////////////////////////////
                
                /////////////// modulator neuron /////////////
                ParameterType modInput = 0;
                ParameterType sourceInput = 0;
                ParameterType ne = netSt->nNSt[modulator].excitation;
                //check if it is excited
                if (checkExcitationThreshold(ne)) {
                    //modulator already activated
                    if (netSt->nNSt[modulator].isFired == true) {
                        modInput = netSt->nNSt[modulator].state;
                    }
                    //modulator not already activated, use the state from the previous iteration
                    else {
                        modInput = netSt->nNSt[modulator].prevState;
                    }
                }
                
                /////////////// source neuron ////////////
                //source already activated
                if (netSt->nNSt[source].isFired == true) {
                    sourceInput = netSt->nNSt[source].state;
                    NConnState cnn = NConnState(connPosition, ctFeedForward);
                    
                    //marking the recurrent connection as used to be processed later on
                    //because we still do not know if this neuron will fire
                    usedConns.push(cnn);
                }
                //recurrent connection
                //source not already activated, use the state from the previous iteration
                else {
                    sourceInput = netSt->nNSt[source].prevState;
                    NConnState cnn = NConnState(connPosition, ctRecurrent);
                    
                    //marking the recurrent connection as used to be processed later on
                    //because we still do not know if this neuron will fire
                    usedConns.push(cnn);
                }
                
                neuronOpr(modInput, sourceInput, sum);
            }
        }
    }
    
    // If this neuron is an input, we need to add the value from
    // the current state to the sum given that later is replaced
    if (neurons[nID].nType == ntInput) {
        sum += netSt->ioVal[nID];
    }
    
    //In the case that the ignore_if_only_recurrent flag is set,
    //the neuron does not activate if it only has recurrent input
    //this is useful for normal "remaining neurons" (neurons that are not control, not input and not output neurons)
    if (onlyRecurrentInputs == true && ignoreIfOnlyRecurrent == true) {
        return 0;
    }
    
    sum = maxWeight(sum);
    
    //mark all feedback connections
    while(!usedConns.empty()) {
        NConnState cnn = usedConns.top();
        usedConns.pop();
        netSt->nCSt[cnn.connID].connType = cnn.connType;
    }
   
    const NNeuron &neuron = neurons[nID];
    ParameterType neuronOutput = activationFunction(neuron.nType, sum);
    netSt->nNSt[nID].state = stateActivation(neuron.firingRate, netSt->nNSt[nID].prevState, neuronOutput);
    
    if (neurons[nID].nType == ntOutput) {
        netSt->ioVal[nID] = neuronOutput;
    }
    
    return neuronOutput;
}

ParameterType NNSFunction::stateActivation(const NFiringRate &nFR, const ParameterType &prevSt, const ParameterType &neuronOutput)
{
    ParameterType beta = betaOfFiringRate(nFR);
    
#ifdef CONTINUOUS_PARAM
    //use Windrow Hoff for slower neurons
    ParameterType state = prevSt + beta * (neuronOutput - prevSt);
#else
    ParameterType state = (prevSt ^ (beta & (~(neuronOutput ^ prevSt))));
#endif
    
    return state;
}

inline ParameterType NNSFunction::activationFunction(const NNeuronType &nt, const ParameterType &weightedInput)
{
    switch(nt)
    {
        case ntRandom:
#ifdef CONTINUOUS_PARAM
            return RandomUtils::randomNormal(0, 1.0);;
#else
            return RandomUtils::randomPositive(MAXIMUM_WEIGHT);
#endif
        case ntActivation: {
#ifdef CONTINUOUS_PARAM
            return tanhf(weightedInput);
#else
            return __builtin_popcount(weightedInput);
#endif
        }
        case ntCPrimer:
        case ntThreshold:
        case ntControl: {
#ifdef CONTINUOUS_PARAM
            return weightedInput >= 0 ? 1.0 : -1.0;
#else
            int val = __builtin_popcount(weightedInput);
            return val >= EXCITATION_THRESHOLD_BITS ? HALF_WEIGHT : 0;
#endif
        }
        case ntRoll: {
#ifdef CONTINUOUS_PARAM
            float change = weightedInput >= 0 ?  1.0 : -1.0;
            return weightedInput + change;
#else
            ushortT val = weightedInput << 1 | weightedInput >> 15;
            return val;
#endif
        }
        case ntID:
        case ntInput:
        case ntOutput: {
            return weightedInput;
        }
        case ntNumberNeuronType:
        default: return 0;
    }
}

inline ParameterType NNSFunction::betaOfFiringRate(const NFiringRate &fr)
{
#ifdef CONTINUOUS_PARAM
    float frD = static_cast<float>(fr);
    float partial = 1 / frD;
    return partial;
#else
    switch(fr) {
        case frL1: return MAXIMUM_WEIGHT;
        case frL7: return 2047;
        case frL49: return 63;
        default: return 0;
    }
#endif
}

// NOTE!! The methods below were initially intended to consider its correxponding
// constants from the program configuration, however, with a simple test, it showed
// a degradation in performance of about 2x, which is not acceptable. Possibly that
// will be supported in future versions.
inline bool NNSFunction::checkExcitationThreshold(const ParameterType &ne)
{
//    PConfig *pConf = PConfig::globalProgramConfiguration();
#ifdef CONTINUOUS_PARAM
//    ParameterType excThr = pConf->excitationThresholdC();
//    return ne >= excThr;
    return ne >= EXCITATION_THRESHOLD;
#else
//    ParameterType excThr = pConf->excitationThresholdB();
//    return ne < excThr;
    return ne < EXCITATION_THRESHOLD;
#endif
}

inline bool NNSFunction::remainingNeuronThreshold(const ParameterType &output)
{
//    PConfig *pConf = PConfig::globalProgramConfiguration();
#ifdef CONTINUOUS_PARAM
//    ParameterType remThr = pConf->remainingNeuronThresholdC();
//    return output > remThr || output < -remThr;
    return output > REMAINING_NEURON_THRESHOLD || output < -REMAINING_NEURON_THRESHOLD;
#else
//    ParameterType remThr = pConf->remainingNeuronThresholdB();
//    return output >= remThr;
    return output >= REMAINING_NEURON_THRESHOLD;
#endif
}

inline ParameterType NNSFunction::maxWeight(const ParameterType &sum)
{
//    PConfig *pConf = PConfig::globalProgramConfiguration();
#ifdef CONTINUOUS_PARAM
//    ParameterType maxW = pConf->maxWeightC();
//    return min(maxW, max(-maxW, sum));
    if (sum < -MAXIMUM_WEIGHT || sum > MAXIMUM_WEIGHT) {
        return MAXIMUM_WEIGHT;
    }
    
    return sum;
#else
//    ParameterType maxW = pConf->maxWeightB();
//    return maxW > sum ? sum : maxW;
    return MAXIMUM_WEIGHT > sum ? sum : MAXIMUM_WEIGHT;
#endif
}

inline void NNSFunction::neuronOpr(const ParameterType &a, const ParameterType &b, ParameterType &c)
{
#ifdef CONTINUOUS_PARAM
    c += a * b;
#else
    c |= a & b;
#endif
}
