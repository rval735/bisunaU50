//
//  UnifiedNeuralModel.hpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 10/6/19.
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

#ifndef UnifiedNeuralModel_hpp
#define UnifiedNeuralModel_hpp

#include <stdio.h>
#include "NN/NNetworkState.hpp"
#include "NN/NNoveltyMap.hpp"
#include "Configuration/PConfig.hpp"

struct UNMCell
{
    NNetworkModule *netMod;
    NNetworkState *netSt;
    float cellFitness;
    uintT cellStep;
    bool deallocate = false;
    UNMCell(NNetworkModule *module = nullptr, NNetworkState *state = nullptr, float fit = 0, uintT step = 0):
        netMod(module), netSt(state), cellFitness(fit), cellStep(step) { };
    UNMCell(ushortT nID, ushortT obs, ushortT actions, ushortT mutations);
    ~UNMCell();
};

struct UNMConfig
{
    ushortT unmObs;
    ushortT unmActions;
    ushortT unmPopulation;
    ushortT unmStepMuts;
    uintT unmGeneration;
    uintT unmSteps;
    uintT unmMapSize;
    
    UNMConfig(const ushortT obs = 0, const ushortT actions = 0,
              const ushortT population = PConfig::globalProgramConfiguration()->populationSize(),
              const ushort stepMutations = PConfig::globalProgramConfiguration()->stepMutations(),
              const uintT generation = 0, const uintT steps = 0,
              const ushortT mapSize = PConfig::globalProgramConfiguration()->noveltyMapSize()):
        unmObs(obs), unmActions(actions), unmPopulation(population), unmStepMuts(stepMutations),
        unmGeneration(generation), unmSteps(steps), unmMapSize(mapSize)
    {};
};

struct UnifiedNeuralModel
{
public:
    UnifiedNeuralModel(const ushortT initialMutations = 0, UNMConfig config = UNMConfig());
    ~UnifiedNeuralModel();
    
    UNMConfig config;
    NNoveltyMap nmap;
    vector<UNMCell *> cells;
};

struct UNMFunctions
{
    static vector<ParameterType> step(const float reward, const vector<ParameterType> observation, UNMCell *cell);
    static ulongT calculateSpectrum(const vector<NNeuron> &neurons);
    static bool compareTwoCells(const UNMCell &a, const UNMCell &b);
    static void modifyCellModule(const ushortT netID, const ushortT stepMut, const NNetworkModule *mod, UNMCell *cell);
    static void noveltyMapParents(const vector<UNMCell *> &agents, NNoveltyMap *nmap);
    static void noveltyPopulationModification(const ushortT stepMut, const NNoveltyMap &nmap, vector<UNMCell *> &cells, vector<ushortT> *prevID = NULL);
    static void spectrumDiversityEvolve(UnifiedNeuralModel *model, vector<ushortT> *lst = NULL);
    static void endCellEpisode(const float reward, const ushortT maxEpisodes, UNMCell *cell);
    static void loadAgent(const char *filename, UnifiedNeuralModel *model);
    static void checkSaveGen(PConfig *pConf, const UnifiedNeuralModel *agent, const char *prefix = NULL, bool encourageSave = false);
    static UnifiedNeuralModel configureModel(ushortT observations, ushortT actions, PConfig *pConf, const char *prefix = NULL);
};

#endif /* UnifiedNeuralModel_hpp */
