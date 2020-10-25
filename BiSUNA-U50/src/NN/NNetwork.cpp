//
//  NNetwork.cpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 14/5/19.
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

#include "NNetwork.hpp"
#include <thread>
#include <climits>
#include "RandomUtils.hpp"
#include "Environments/ReinforcementEnvironment.hpp"
#include "NN/NNetworkExtra.hpp"
#include "OCL/OCLContainer.hpp"
#include "NN/NNetworkOCL.hpp"

float NNFunction::mainBiSUNA(const char *config)
{
    PConfig::discardConfiguration();
    PConfig *pConf = PConfig::globalProgramConfiguration(config);
    printf("Config file: %s \n", config);

#ifdef CONTINUOUS_PARAM
    printf("Using continuous neurons\n");
#else
    printf("Using binary neurons\n");
#endif

#ifdef DEBUG
    if (pConf->enableDebugging()) {
        // This helps to keep consistency across execution and testing
        // setting the seed to an specific value.
        ushortT rndSeed = pConf->randomSeed();
        RandomUtils::changeRandomSeed(rndSeed);
    }
#endif // DEBUG

    switch (pConf->exeType()) {
        case PCGThread: {
        	printf("Thread mode: %i\n", pConf->threadNumber());
            return NNFunction::mainThreadedNNetwork(pConf);
            break;
        }
        case PCGOpenCL: {
        	printf("OpenCL mode, device: %i\n", pConf->deviceType());
            return NNFunctionOCL::mainOCLNetwork(pConf);
            break;
        }
        case PCGOpenCLFull:
            return 0;
            break;
        default:
        	return 0;
        	break;
    }
}

float NNFunction::mainThreadedNNetwork(PConfig *pConf)
{
    ushortT population = pConf->populationSize();
    uintT generations = pConf->generations();
    ushortT episodesPerAgent = pConf->episodesPerAgent();
    ushortT numThreads = pConf->threadNumber();
    ushortT cellsPerThread = population / numThreads;

    vector<ReinforcementEnvironment *> envs = RLFunctions::environmentVector(numThreads, pConf);
    int numberObsVars = envs[0]->observationVars;
    int numberActionVars = envs[0]->actionVars;

    UnifiedNeuralModel agent = UNMFunctions::configureModel(numberObsVars, numberActionVars, pConf);
    uintT maxGens = agent.config.unmGeneration + generations;
    vector<float> rewards(numThreads);

    auto envSetFunc = [&envs]() {
        uintT nSeed = RandomUtils::randomUInt(UINT_MAX);
        envs[0]->set(nSeed);
    };

    envSetFunc();

    auto trFunc = [&agent, &envs, &numberObsVars, &episodesPerAgent]
        (ushortT thrID, ushortT start, ushortT end, float *bestReward) {
//        printf("Thread: % i, Start: %i, End: %i\n", thrID, start, end);
            ReinforcementEnvironment *env = envs[thrID];
            float reward = env->step(NULL);
            int stepCounter = 0;
            int i = env->trial;
            ushortT current = start;
            float bestSoFar = EXTREME_NEGATIVE_REWARD;
            while (current <= end) {
                float accumReward = reward;
                UNMCell *cell = agent.cells[current];

                while(env->trial == i && stepCounter < env->maxSteps) {
                    vector<ParameterType> obs(env->observation, env->observation + numberObsVars);
                    vector<ParameterType> action = UNMFunctions::step(reward, obs, cell);
                    reward = env->step(action.data());
                    accumReward += reward;
                    stepCounter++;
//                    printf("%i Obs: %.0f, Act: %.4f Reward: %f\n", stepCounter, env->observation[0], action[0], reward);
                }

                if (accumReward > bestSoFar) {
                    bestSoFar = accumReward;
//                    printf("Thread %i best update, agent: %i, Gen: %i, Fitness: %f\n",
//                           thrID, current, agent.config.unmGeneration, bestSoFar);
                }

                UNMFunctions::endCellEpisode(reward, episodesPerAgent, cell);

                if (env->trial == i) {
                    reward = env->restart();
                }
                else {
                    reward = env->step(NULL);
                }

                stepCounter = 0;
                i = env->trial;
                current++;
            }

//            printf("Thread %i best: %f\n", thrID, bestSoFar);
            *bestReward = bestSoFar;
        };

    while (agent.config.unmGeneration < maxGens) {
        printf("Generation: %i of %i\n", agent.config.unmGeneration, maxGens);
        vector<thread> trds;
        for (int i = 0; i < numThreads; i++) {
            trds.push_back(thread(
                        trFunc,
                        i,
                        i * cellsPerThread,
                        i * cellsPerThread + cellsPerThread - 1,
                        &(rewards[i])
                        ));
        }

        for_each(trds.begin(), trds.end(), [](thread &t){
            t.join();
        });

        sort(rewards.rbegin(), rewards.rend());
        printf("From all threads, best was: %f\n", rewards[0]);

        UNMFunctions::checkSaveGen(pConf, &agent);

        UNMFunctions::spectrumDiversityEvolve(&agent);

        envSetFunc();

        printf("----------------------------------------\n");
    }

    if (pConf->saveToFile()) {
        NNExFunction::writeBiSUNAModel(&agent, pConf->bisunaFile().c_str());
    }

    for_each(envs.begin(), envs.end(), [](ReinforcementEnvironment *env){
        delete env;
    });

    return rewards[0];
}
