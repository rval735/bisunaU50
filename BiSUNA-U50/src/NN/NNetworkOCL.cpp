//
//  NNetworkOCL.cpp
//  BiSUNAOpenCL
//
//  Created by RH VT on 11/03/20.
//  Copyright © 2020 R. All rights reserved.
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

#include "NNetworkOCL.hpp"
#include "RandomUtils.hpp"

// #define BREAKEARLY // This tag is used to break execution early for testing purposes only

void printCell(CLCell *cell, ushort size)
{
    printf("---////////////---\n");
    for (ushort i = 0; i < size; i++) {
        CLNeuron *nrs = cell[i].clNrs;
        CLConnection *cons = cell[i].clCons;
        ushort nrsSize = cell[i].nrsSize;
        ushort connSize = cell[i].connSize;
        printf("CLCell %i, Nrs %i [\n", cell[i].clNetworkID, nrsSize);
        CLNeuron nrn = nrs[0];
        CLConnection conn = cons[0];
        for (ushort j = 0; j < nrsSize; j++) {
#ifdef CONTINUOUS_PARAM
            printf("%i\t%i\t%f\t%f\t%f\t%i\n", nrn.clNID, nrn.clFRNT, nrn.clSt, nrn.clPSt, nrn.clExc, nrn.clFired);
#else
            printf("%i\t%i\t%i\t%i\t%i\t%i\n", nrn.clNID, nrn.clFRNT, nrn.clSt, nrn.clPSt, nrn.clExc, nrn.clFired);
#endif
            nrn = nrs[j+1];
        }
        printf("]\nCons %i [\n", connSize);
        for (ushort j = 0; j < connSize; j++) {
            printf("%i\t%i\t%i\t%i\t%i\n", conn.clFromNID, conn.clToNID, conn.clNeuroMod, conn.clCnType, conn.clPCnType);
            conn = cons[j+1];
        }
        printf("]===\\\\\\\\\\\\===\n");
    }
}

float NNFunctionOCL::executeOCL(vector<ReinforcementEnvironment *>env,
                                const ushort &episodesPerAgent,
                                OCLContainer *cont,
                                bool profiling)
{
    size_t envSize = env.size();
    float *reward = cont->rewardsRef;
    vector<pair<int, int>> envCounter(envSize); // first counts the current trial, the second counts the number of steps
    vector<bool> stillExecuting(envSize, true);
    
    for(uintT i = 0; i  < envSize; i++) {
        reward[i] = env[i]->step(NULL);
        envCounter[i] = {env[i]->trial, 0};
    }
    
    float bestSoFar = EXTREME_NEGATIVE_REWARD;
    ushort obsSize = CELL_POPULATION * INPUT_SIZE;
    ushort actSize = OUTPUT_SIZE;
    bool shouldContinue = false;
    ushortT counter = 0;
    double profTaskAccum = 0;
    function<void(double)> profCB = [&counter, &profTaskAccum](double taskP) {
        counter++;
//        printf("Counter: %i, Task: %f, Output: %f\n", counter, taskP, outputP);
        profTaskAccum += taskP;
    };
    
    if (!profiling) {
        profCB = NULL;
    }
    
    long numCU = cont->computeUnits;
    
    do {
        ushort obsCounter = 0;
        ushort obsIdx = 0;
        
        ParameterType *ref = cont->inRef;
        
        for (ushortT i = 0; i < obsSize; i++) {
            ref[i] = env[obsIdx]->observation[obsCounter];
            obsCounter++;
            if (obsCounter >= env[obsIdx]->observationVars) {
                obsIdx++;
                obsCounter = 0;
            }
        }
        
        cont->oclRt->queue.enqueueMigrateMemObjects({cont->inBuffer}, 0);
//        printCell(cont->clUNM->cells, 1);
        int mcuCounter = 0;
        do {
			cont->exeOCL(cont->kernelState[mcuCounter], profCB);
			mcuCounter++;
		} while (mcuCounter < numCU);

        profTaskAccum = 0; // Because profCB is shared between execute and read, it resets here its value.
        cl_int err = cont->oclRt->queue.enqueueMigrateMemObjects({cont->outBuffer}, CL_MIGRATE_MEM_OBJECT_HOST);
        oclCheckError(err, "Error at enqueueMigrateMemObjects for outBuffer\n");
        err = cont->oclRt->queue.finish();
        oclCheckError(err, "Error while waiting for queue to finish\n");

//        printCell(cont->clUNM->cells, 1);
        
        for (ushortT i = 0; i < env.size(); i++) {
            ushortT base = i * actSize;
            ParameterType *actVec = cont->outRef + base;
            
            if (envCounter[i].first == env[i]->trial) {
                reward[i] += env[i]->step(actVec);
                envCounter[i].second++;
            }
        }
        
        shouldContinue = false;
        for (ushort i = 0; i < env.size(); i++ && shouldContinue == false) {
            int trial = env[i]->trial;
            int maxStps = env[i]->maxSteps;
            auto envCnt = envCounter[i];
            // Only when all environments have reached a max number of steps or the trial moved on, it is
            // when this flag will be set to false.
            shouldContinue = shouldContinue || (envCnt.first == trial && envCnt.second <= maxStps);
        }
        
////         TODO!! Remove line below, only for hardware simulation
#ifdef BREAKEARLY
        shouldContinue = false;
#endif
//        printf("%i Obs: %.0f, Act: %.4f Reward: %f\n", stepCounter, env->observation[0], action[0], reward);
    } while(shouldContinue);

    for(uintT i = 0; i < envSize; i++) {
        float curr = reward[i];
        if (curr > bestSoFar) {
            bestSoFar = curr;
        }
    }
    
    printf("Best Fitness: %f\n", bestSoFar);
    
    if (cont->kernelEpisodeEnabled == true) {
        cont->exeOCL(cont->kernelEpisode, NULL);
    }
    else {
        CLCell *cell = cont->cellBufferMap;
        for(uintT i = 0; i < envSize; i++) {
            cell[i].clFitness = reward[i];
            cell[i].cellStep = 0;
            for (uintT j = 0; j < cell[i].nrsSize; j++) {
                cell[i].clNrs[j].clFired = false;
                cell[i].clNrs[j].clSt = 0;
                cell[i].clNrs[j].clExc = 0;
                cell[i].clNrs[j].clPSt = 0;
            }
            
            for (uintT j = 0; j < cell[i].connSize; j++) {
                cell[i].clCons[j].clCnType = false;
                cell[i].clCons[j].clPCnType = false;
            }
        }
    }

    for (ushort i = 0; i < env.size(); i++) {
        reward[i] = env[i]->restart();
        envCounter[i].first = env[i]->trial;
        envCounter[i].second = 0;
    }
    
    if (counter != 0) {
        printf("Counter %i, TaskSum: %0.1f (%0.1f)\n", counter, profTaskAccum, profTaskAccum / counter);
    }

//    printf("Thread %i best: %f\n", thrID, bestSoFar);
    return bestSoFar;
}

void copyUNMToOCL(ushortT population, OCLContainer *cont, UnifiedNeuralModel *unm)
{
    for (uintT i = 0; i < population; i++) {
        unm->cells[i]->cellFitness = cont->cellBufferMap[i].clFitness;
    }

    UNMFunctions::spectrumDiversityEvolve(unm);
    CLUnifiedNeuralModel localUNM = OCLBridge::toCLUNM(*unm);
    memcpy(cont->cellBufferMap, &(localUNM.cells), sizeof(CLCell) * population);
    cont->oclRt->queue.enqueueMigrateMemObjects({cont->unmBuffer}, 0);
}

float NNFunctionOCL::mainOCLNetwork(PConfig *pConf)
{
    ushortT population = pConf->populationSize();
    ushortT generations = pConf->generations();
    ushortT genCounter = 0;
    ushortT episodesPerAgent = pConf->episodesPerAgent();
    vector<ReinforcementEnvironment *> envs = RLFunctions::environmentVector(population, pConf);
    int numberObsVars = envs[0]->observationVars;
    int numberActionVars = envs[0]->actionVars;
    bool profiling = pConf->oclProfiling();
    UnifiedNeuralModel unm = UNMFunctions::configureModel(numberObsVars, numberActionVars, pConf);
    OCLContainer *cont = new OCLContainer(pConf, &unm, NULL);
    bool endEpiOCL = cont->kernelEpisodeEnabled;
    bool exeEvoOCL = cont->kernelEvolvedEnabled;
    endEpiOCL ? cont->loadParametersKrnEpisode() : void();
    exeEvoOCL ? cont->loadParametersKrnEvolve() : void();
    
    ushortT everyNGens = pConf->saveEveryNGenerations();
    bool saveToFile = pConf->saveToFile();
    bool saveByGens = false;
    float bestInGen = 0;
    
    while (genCounter < generations) {
        printf("Generation: %i\n", genCounter);
        bestInGen = NNFunctionOCL::executeOCL(envs, episodesPerAgent, cont, profiling);
        saveByGens = genCounter % everyNGens == 0;
        
        if (exeEvoOCL == true) {
            cont->exeOCL(cont->kernelEvolve);
            
            if (saveToFile && saveByGens) {
                UnifiedNeuralModel unm = OCLBridge::fromCLUNM(*(cont->clUNM));
                UNMFunctions::checkSaveGen(pConf, &unm);
            }
        }
        else {
            copyUNMToOCL(population, cont, &unm);
            saveToFile && saveByGens ? UNMFunctions::checkSaveGen(pConf, &unm) : void();
        }
        
        genCounter++;
////        TODO!! Remove line below, only for hardware simulation
#ifdef BREAKEARLY
        genCounter = OUT_INDEX;
#endif
        printf("----------------------------------------\n");
    }
    
    if (saveToFile) {
//        UnifiedNeuralModel unm = OCLBridge::fromCLUNM(*(cont->clUNM));
        NNExFunction::writeBiSUNAModel(&unm, pConf->bisunaFile().c_str());
    }
    
    for_each(envs.begin(), envs.end(), [](ReinforcementEnvironment *env){
        delete env;
    });
    
    if (cont->devType != CL_DEVICE_TYPE_ACCELERATOR) {
        delete cont;
    }
    
    return bestInGen;
}
