//
//  NNetworkState.hpp
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

#ifndef NNetworkState_hpp
#define NNetworkState_hpp

#include "NNetworkModule.hpp"
#include <unordered_map>

struct NNetworkState
{
public:
    NNetworkState(NNetworkModule *ptrModule = nullptr);
    ~NNetworkState();
    
    vector<NNeuronState> nNSt;
    vector<NConnState> nCSt;
    unordered_map<ushortT, ParameterType> ioVal;
    NNetworkModule *module;
};

struct NNSFunction
{
    static void resetState(NNetworkState *st);
    static void processInputNeurons(const vector<ParameterType> &input, NNetworkState *netSt);
    static vector<ParameterType> processOutputNeurons(NNetworkState *netSt);
    static void processCPrimerNeurons(NNetworkState *netSt);
    static void processControlNeurons(NNetworkState *netSt);
    static void processRemainingNeurons(NNetworkState *netSt);
    static void runControlID(const ushortT controlID, const vector<NConnection> &conns, vector<NNeuronState> *neuronsSt);
    static vector<ParameterType> process(const vector<ParameterType> &input, NNetworkState *netSt);
    static ParameterType execute(const ushortT &nID, NNetworkState *netSt, const bool &ignoreIfOnlyRecurrent = false);
    static ParameterType stateActivation(const NFiringRate &nFR, const ParameterType &prevSt, const ParameterType &neuronOutput);
    static ParameterType activationFunction(const NNeuronType &nt, const ParameterType &weightedInput);
    static ParameterType betaOfFiringRate(const NFiringRate &fr);
    static bool checkExcitationThreshold(const ParameterType &ne);
    static bool remainingNeuronThreshold(const ParameterType &output);
    static ParameterType maxWeight(const ParameterType &sum);
    static void neuronOpr(const ParameterType &a, const ParameterType &b, ParameterType &c);
};
#endif /* NNetworkState_hpp */
