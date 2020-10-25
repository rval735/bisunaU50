//
//  ReinforcementEnvironment.hpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 22/7/19.
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

#ifndef ReinforcementEnvironment_hpp
#define ReinforcementEnvironment_hpp

#include "Configuration/PConfig.hpp"

//features (in fact, they are restrictions to the reinforcement problem)
//	NormalizedObservation - Observation real values must be in the [-1,1] range
//	NormalizedAction - Action real values are supposed to be in the [0,1] range
enum NomalizedValues {
    NormalizedObservation,
    NormalizedAction
};

struct ReinforcementEnvironment
{
    ReinforcementEnvironment(ushortT eID, const char *fileName);
    virtual ~ReinforcementEnvironment();
    
    ParameterType *observation;
    int observationVars;
    int actionVars;
    int trial;
    int maxSteps;
    
    ushortT envID;
    INIReader ini;
    
    virtual void start(int &numObsVars, int &numActionVars) = 0;
    virtual float step(ParameterType *action) = 0;
    virtual float restart() = 0;
    virtual void print() { };
    virtual bool set(int feature) { return false; }; //see enum above for the acceptable features
};

struct RLFunctions
{
    static ReinforcementEnvironment *createEnvFromStr(const ushortT &envID, const PCEnvironmentSupported &env, const string &envConf);
    static vector<ReinforcementEnvironment *> environmentVector(const ushortT numEnv, PConfig *pConf);
};

#endif
