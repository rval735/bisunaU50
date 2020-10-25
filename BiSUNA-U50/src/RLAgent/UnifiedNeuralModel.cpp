//
//  UnifiedNeuralModel.cpp
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

#include "UnifiedNeuralModel.hpp"
#include "NN/NNetworkModule.hpp"
#include "RandomUtils.hpp"
#include "NN/NNetworkExtra.hpp"
#include <algorithm>
#include <unordered_set>
#include <cassert>
#include <unordered_set>

/////////////////////////////////////////////////
//////////////////// UNMCell ////////////////////
/////////////////////////////////////////////////
UNMCell::UNMCell(ushortT nID, ushortT obs, ushortT actions, ushortT mutations)
{
    netMod = new NNetworkModule(nID, obs, actions);
    
    for (ushortT j = 0; j < mutations; j++) {
        netMod->structuralMutation();
    }
    
    // Primer control neurons can only have outgoing connections
    // for that reason, it is neccessary to change it to control.
    // Later on, structural mutation will check for ntControl
    // neurons that do not receive connections and can
    // act as ntCPrimer.
    NNMFunction::updatePrimers(netMod->nConns, &(netMod->nNeurons));
    
    // Initialize state
    netSt = new NNetworkState(netMod);
    cellFitness = 0;
    cellStep = 0;
    deallocate = true;
}

UNMCell::~UNMCell()
{
    if (deallocate) {
        delete netSt;
        delete netMod;
    }
}

////////////////////////////////////////////////////////////
//////////////////// UnifiedNeuralModel ////////////////////
////////////////////////////////////////////////////////////
UnifiedNeuralModel::UnifiedNeuralModel(const ushortT initialMutations, const UNMConfig config):
    config(config)
{
    unordered_set<ulongT> spectrum;
    
    // If it failed here, it means the population size must be larger than novelty map
    assert(config.unmMapSize < config.unmPopulation);
    
    for (ushortT i = 0; i < config.unmPopulation; i++) {
        UNMCell *cell = new UNMCell(i, config.unmObs, config.unmActions, initialMutations);
        // Calculate the spectrum that is going to be used later to create
        // the novelty map
        ulongT spec = UNMFunctions::calculateSpectrum(cell->netMod->nNeurons);
        spectrum.emplace(spec);
        
        cells.push_back(cell);
    }
    
    // From all spectrums, obtain just enough that can fill the novelty map size
    ushortT mapSize = config.unmMapSize;
    vector<ulongT> specs;
    specs.reserve(mapSize);
    
    // Transform the set into a vector that has all unique spectrum values.
    for (auto i = spectrum.begin(); i != spectrum.end() && mapSize > 0; i++) {
        specs.push_back((*i));
        mapSize--;
    };
    
    // In case there are not enough cells with sufficiently different spectrum, it needs to fill
    // those spaces with randomly selected copies.
    while (specs.size() < config.unmMapSize) {
        ushort pos = RandomUtils::randomPositive(config.unmPopulation - 1);
        ulongT spec = UNMFunctions::calculateSpectrum(cells[pos]->netMod->nNeurons);
        specs.push_back(spec);
    }
    
    // Construct the novelty map with unique spectrums
    nmap = NNoveltyMap(specs);
}

UnifiedNeuralModel::~UnifiedNeuralModel()
{
    for_each(cells.begin(), cells.end(), [](UNMCell *cell){
        delete cell;
    });
}

//////////////////////////////////////////////////////
//////////////////// UNMFunctions ////////////////////
//////////////////////////////////////////////////////
// In order to create the spectrum of a NNeuron vector, a single 64 bits unsigned
// integer is needed, given that there are 6 different types of categories, each
// uses 10 bits of it, the schematic is as follows:
// ntID         0               - 1023
// ntThreshold  1024            - 1047552
// ntRandom     1048576         - 1072693248
// ntControl    1073741824      - 1098437885952
// ntActivation 1099511627776   - 1124800395214848
// firingRate   1125899906842624- 1151795604700004352
// ntCPrimer 1152921504606846976- 17293822569102704640
// Using 64 bits, the network spectrum can differentiate neuron types up to
// 1023 of the same type and combined 7161 within the spectra. If more types are
// needed, then reduce the bit size of 10 or use two or more ulongT to store data.
inline ulongT UNMFunctions::calculateSpectrum(const vector<NNeuron> &neurons)
{
    ulongT tID = 0, tTh = 0, tRa = 0, tCo = 0, tAc = 0, tFR = 0, tCP = 0;
    
    for (auto i = neurons.begin(); i != neurons.end(); i++) {
        switch ((*i).nType) {
            case ntID:
                tID++;
                break;
            case ntThreshold:
                tTh++;
                break;
            case ntRandom:
                tRa++;
                break;
            case ntControl:
                tCo++;
                break;
            case ntActivation:
                tAc++;
                break;
            case ntCPrimer:
            case ntRoll:
                tCP++;
                break;
            default:
                break;
        }
        
        if ((*i).firingRate != frL1) {
            tFR++;
        }
    }
    
    tID = tID < 1023 ? tID & 1023 : 1023;
    tTh = tTh < 1023 ? (tTh << 10) & 1047552 : 1047552;
    tRa = tRa < 1023 ? (tRa << 20) & 1072693248 : 1072693248;
    tCo = tCo < 1023 ? (tCo << 30) & 1098437885952 : 1098437885952;
    tAc = tAc < 1023 ? (tAc << 40) & 1124800395214848 : 1124800395214848;
    tFR = tFR < 1023 ? (tFR << 50) & 1151795604700004352 : 1151795604700004352;
    // Primer neurons could only be counted up to 15
    ulongT maxtCP = 0xF000000000000000;
    tCP = tCP < 15 ? (tCP << 60) & maxtCP : maxtCP;
    
    return tID | tTh | tRa | tCo | tAc | tFR | tCP;
}

// Given two UNMCell, it will compare multiple properties to return
// the one that fits all characteristics: better fitness or same
// fitness with less neurons. If the first is better, it will return
// true, false otherwise
bool UNMFunctions::compareTwoCells(const UNMCell &a, const UNMCell &b)
{
    bool betterFitness = a.cellFitness > b.cellFitness;
    bool lessEqNeurons = a.netMod->nNeurons.size() <= b.netMod->nNeurons.size();
    bool sameFitness = a.cellFitness == b.cellFitness;
    
    return betterFitness || (sameFitness && lessEqNeurons);
}

void UNMFunctions::noveltyMapParents(const vector<UNMCell *> &agents, NNoveltyMap *nmap)
{
    size_t agentSize = agents.size();
    // From all agents, decide which are going to be selected as parents
    for(size_t i = 0; i < agentSize; i++) {
        UNMCell *cell = agents[i];
        // Calculate each agent spectrum
        ulongT spectrum = calculateSpectrum(cell->netMod->nNeurons);
        
        // Compare the spectrum to what the in Novelty Map currently holds, if it
        // is considered a "novel" spectrum, then it will store that value and
        // return the "wost" individual given the distance metric. In case it is
        // not inserted, it will return the closest individual to what the nmap
        // currently holds.
        ushortT idxPos = NNMapFunction::idxPosition(spectrum, nmap);
        NMStr str = nmap->nmStrs[idxPos];
        
        if (str.ptr == NULL) {
            nmap->nmStrs[idxPos].ptr = (void *)cell;
            continue;
        }
        
        UNMCell *stored = (UNMCell *)str.ptr;
        // This case, it means that the agent spectra is not substantially different
        // from what the novelty map currently holds, it is necessary to compare
        // more details about it
        bool isCurrentBetter = compareTwoCells(*cell, *stored);
        
        // In case that the current agent has a better fitness than the one
        // referenced by the nmap, it will replace its stored pointer.
        if (isCurrentBetter) {
            nmap->nmStrs[idxPos].ptr = (void *)cell;
        }
    }
}

void UNMFunctions::modifyCellModule(const ushortT netID, const ushortT stepMut, const NNetworkModule *mod, UNMCell *cell)
{
    cell->cellFitness = 0;
    cell->cellStep = 0;
    NNetworkModule *toUpdate = cell->netMod;
    toUpdate->nNeurons = mod->nNeurons;
    toUpdate->nConns = mod->nConns;
    toUpdate->nParams.networkID = netID;

    for (int k = 0; k < stepMut; ++k) {
        toUpdate->structuralMutation();
    }
    
    NNMFunction::updatePrimers(toUpdate->nConns, &(toUpdate->nNeurons));
    
    for_each(toUpdate->nConns.begin(), toUpdate->nConns.end(), [](NConnection &conn){
        NNMFunction::weightMutation(&conn);
    });
    
    ushortT connSize = toUpdate->nConns.size();
    cell->netSt->nNSt.resize(toUpdate->nNeurons.size());
    cell->netSt->nCSt.resize(connSize);
    NNSFunction::resetState(cell->netSt);
    
    for (int i = 0; i < connSize; i++) {
        NConnState &st = cell->netSt->nCSt[i];
        st.connID = i;
    };
}

void UNMFunctions::noveltyPopulationModification(const ushortT stepMut, const NNoveltyMap &nmap, vector<UNMCell *> &cells, vector<ushortT> *prevID)
{
    ushortT cellsSize = cells.size();

    // If it crashed here, it means that the UNMCell is not > 1
    assert(cellsSize > 1);
    
    for (ushortT i = 0; i < nmap.popSize; i++) {
        UNMCell *nmapCell = (UNMCell *)nmap.nmStrs[i].ptr;
        
        while (nmapCell == NULL) {
            ushortT randIdx = RandomUtils::randomPositive(nmap.popSize - 1);
            nmapCell = (UNMCell *)nmap.nmStrs[randIdx].ptr;
        }
        
        ushortT nID = nmapCell->netMod->nParams.networkID;
        
        if (cells[i]->netMod->nParams.networkID != nID) {
            UNMCell *ptr = cells[i];
            cells[i] = cells[nID];
            cells[nID] = ptr;
        }
        
        // Even if it is a parent from the novelty map or any other
        // agent, it needs to reset the fitness value
        cells[i]->cellFitness = 0;
        cells[i]->cellStep = 0;
        
        if (prevID != NULL) {
            prevID->push_back(nID);
        }
        
        cells[i]->netMod->nParams.networkID = i;
        NNSFunction::resetState(cells[i]->netSt);
    }
    
    for (ushortT i = nmap.popSize; i < cellsSize; i++) {
        // Randomly choose one element from the novelty map size, because all
        // selected parents are now within the first elements in cells,
        // they can be referenced from there.
        ushortT randIdx = RandomUtils::randomPositive(nmap.popSize - 1);
        // Identify which is the reference the novelty map
        // relating to the parent index
        NNetworkModule *mod = cells[randIdx]->netMod;
        // Modify current cell with the values of the parent
        // it means the id the new cell will hold, mod from
        // which cell it is going to be replicated and cells[i]
        // where it is going to be modified
        modifyCellModule(i, stepMut, mod, cells[i]);
    }
}

void UNMFunctions::spectrumDiversityEvolve(UnifiedNeuralModel *model, vector<ushortT> *lst)
{
    UNMConfig &conf = model->config;
    NNoveltyMap &nmap = model->nmap;
    vector<UNMCell *> &cells = model->cells;
    
    noveltyMapParents(cells, &nmap);
    noveltyPopulationModification(conf.unmStepMuts, nmap, cells, lst);
    
    for (NMStr &str : nmap.nmStrs) {
        // After finishing using the nmap, the stored pointer
        // should be nullified, however, the values will remain
        // stored in the nmap
        str.ptr = NULL;
    }
    
    for_each(cells.begin(), cells.end(), NNExFunction::removeOrphans);
    
    conf.unmGeneration++;
    conf.unmSteps = 0;
}

vector<ParameterType> UNMFunctions::step(const float reward, const vector<ParameterType> observation, UNMCell *cell)
{
    vector<ParameterType> output = NNSFunction::process(observation, cell->netSt);
    cell->cellFitness += reward;
    return output;
}

void UNMFunctions::endCellEpisode(const float reward, const ushortT maxEpisodes, UNMCell *cell)
{
    cell->cellFitness += reward;
    cell->cellStep += 1;
    
    if (cell->cellStep >= maxEpisodes) {
        cell->cellFitness /= cell->cellStep;
        cell->cellStep = 0;
    }
    else {
        return;
    }
    
    NNSFunction::resetState(cell->netSt);
}

void UNMFunctions::loadAgent(const char *filename, UnifiedNeuralModel *model)
{
    NNetworkModule modInt = NNExFunction::loadNetworkModule(filename);
    
    int i = 0;
    NNetworkModule *mod = new NNetworkModule(modInt);
    mod->nParams.networkID = i;
    NNetworkState *st = new NNetworkState(mod);
    UNMCell *cell = new UNMCell(mod, st);
    delete model->cells[i]->netMod;
    delete model->cells[i]->netSt;
    delete model->cells[i];
    model->cells[i] = cell;
    
    ulongT w = calculateSpectrum(modInt.nNeurons);
    vector<ulongT> specs(model->config.unmMapSize, w);
    model->nmap = NNoveltyMap(specs);
}

void UNMFunctions::checkSaveGen(PConfig *pConf, const UnifiedNeuralModel *agent, const char *prefix, bool encourageSave)
{
    bool saveToFile = pConf->saveToFile();
    
    if (saveToFile == false) {
        return;
    }
    
    ushortT everyNGens = pConf->saveEveryNGenerations();
    bool saveByGens = agent->config.unmGeneration % everyNGens == 0 ;
    bool shouldSave = encourageSave || saveByGens;
    if (shouldSave == true) {
        string bisunaName = pConf->bisunaFile();
        size_t indexPoint = bisunaName.find_last_of(".");
        string genTag = "Gen" + to_string(agent->config.unmGeneration) + "Time";
        bisunaName.insert(indexPoint, genTag);
        if (prefix != NULL) {
            bisunaName.insert(0, prefix);
        }
        
        string timeStamped = NNExFunction::appendTimeStamp(bisunaName);
        NNExFunction::writeBiSUNAModel(agent, timeStamped.c_str());
    }
}

UnifiedNeuralModel UNMFunctions::configureModel(ushortT observations, ushortT actions, PConfig *pConf, const char *prefix)
{
    ushortT population = pConf->populationSize();
    
    UNMConfig conf = UNMConfig(observations, actions, population, pConf->stepMutations());
    UnifiedNeuralModel agent = UnifiedNeuralModel(pConf->initialMutations(), conf);

    // This function will load a SUNA structured file into the first cell un the UNM
    if (pConf->loadFromFile()) {
        string fileName = pConf->bisunaFile();
        if (prefix != NULL) {
            fileName.insert(0, prefix);
        }
        NNExFunction::readBiSUNAModel(fileName.c_str(), &agent);
    }
    
    return agent;
}

