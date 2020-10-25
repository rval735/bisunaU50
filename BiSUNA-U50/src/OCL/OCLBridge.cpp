//
//  OCLBridge.cpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 2/8/19.
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

#include "OCLBridge.hpp"

ushortT OCLBridge::toFRNT(const NFiringRate &fr, const NNeuronType &nt)
{
    ushortT frU = 0;

    switch (fr) {
        case frL1: frU = 0; break;
        case frL7: frU = 1; break;
        case frL49: frU = 2; break;
        default: frU = frNumberFiringRate;
    }
    
    ushortT nTypeU = ushortT(nt);
    ushortT res = nTypeU | (frU << 11);
    return res;
}

pair<NFiringRate, NNeuronType> OCLBridge::fromFRNT(const ushortT &frNT)
{
    ushortT fr = frNT >> 11;
    switch (fr) {
        case 0: fr = 1; break;
        case 1: fr = 7; break;
        case 2: fr = 49; break;
        default: fr = 3;
    }
    
    ushortT nType = frNT & 2047;
    return {NFiringRate(fr), NNeuronType(nType)};
}

// A NNeuron is transformed into a two element vector, with the
// following characteristics:
// idx 0 = Neuron ID;
// idx 1 = 000000 - 0000000000
// Idx 1 is composed of neuronType (bits 0 - 10) and firing rate
// (bits 11 - 16). If the firing rate is needed, it can simply be
// shifted 10 spaces right and the result would give the FR.
CLNeuron OCLBridge::toCLNeuron(const NNeuron &n, const NNeuronState &nSt)
{
    CLNeuron clN;
    clN.clNID = n.nID;
    clN.clFRNT = toFRNT(n.firingRate, n.nType);
    clN.clSt = nSt.state;
    clN.clPSt = nSt.prevState;
    clN.clExc = nSt.excitation;
    clN.clFired = nSt.isFired;
    return clN;
}

pair<NNeuron, NNeuronState> OCLBridge::fromCLNeuron(const CLNeuron &clN)
{
    NNeuron n = NNeuron();
    n.nID = clN.clNID;
    auto frNT = fromFRNT(clN.clFRNT);
    n.firingRate = frNT.first;
    n.nType = frNT.second;
    
    NNeuronState nSt = NNeuronState();
    nSt.state = clN.clSt;
    nSt.prevState = clN.clPSt;
    nSt.excitation = clN.clExc;
    nSt.isFired = clN.clFired;
    
    return make_pair(n, nSt);
}

CLConnection OCLBridge::toCLConnection(const NConnection &c, const NConnState &cst)
{
    CLConnection clC = CLConnection();
    clC.clFromNID = c.fromNID;
    clC.clToNID = c.toNID;
    clC.clNeuroMod = c.neuroMod;
    clC.clWeight = c.weight;
    clC.clCnType = cst.connType != ctRecurrent;
    clC.clPCnType = cst.prevConnType != ctRecurrent;
    
    return clC;
}

pair<NConnection, NConnState> OCLBridge::fromCLConnection(const ushortT &cID, const CLConnection &clC)
{
    NConnection c = NConnection();
    c.fromNID = clC.clFromNID;
    c.toNID = clC.clToNID;
    c.neuroMod = clC.clNeuroMod;
    c.weight = clC.clWeight;
    
    NConnState cst = NConnState();
    cst.connID = cID;
    cst.connType = clC.clCnType ? ctFeedForward : ctRecurrent;
    cst.prevConnType = clC.clPCnType ? ctFeedForward : ctRecurrent;
    
    return {c, cst};
}

CLUnifiedNeuralModel OCLBridge::toCLUNM(const UnifiedNeuralModel &unm)
{
    OCLBridge::checkConstantsMatch(unm);
    
    CLUnifiedNeuralModel clUNM;
    clUNM.clGeneration = unm.config.unmGeneration;
    clUNM.clSteps = unm.config.unmSteps;
    clUNM.nmap.lastIdx.ciDist = unm.nmap.lastIdx.ciDist;
    clUNM.nmap.lastIdx.ciIndex = unm.nmap.lastIdx.ciIndex;
    
    for (uintT i = 0; i < NMSIZE; i++) {
        clUNM.nmap.nmStrs[i].weight = unm.nmap.nmStrs[i].weight;
        UNMCell *cell = (UNMCell *)unm.nmap.nmStrs[i].ptr;
        if (cell != NULL) {
            clUNM.nmap.nmStrs[i].cellRef = cell->netMod->nParams.networkID;
        }
        else {
            clUNM.nmap.nmStrs[i].cellRef = OUT_INDEX;
        }
    }

    for (uintT i = 0; i < CELL_POPULATION; i++) {
        UNMCell *cell = unm.cells[i];
        clUNM.cells[i].cellStep = cell->cellStep;
        clUNM.cells[i].clFitness = cell->cellFitness;
        clUNM.cells[i].clNetworkID = cell->netMod->nParams.networkID;
        clUNM.cells[i].nrsSize = cell->netMod->nNeurons.size();
        clUNM.cells[i].connSize = cell->netMod->nConns.size();
        
        for (uintT j = 0; j < clUNM.cells[i].nrsSize; j++) {
            CLNeuron clN = toCLNeuron(cell->netMod->nNeurons[j], cell->netSt->nNSt[j]);
            clUNM.cells[i].clNrs[j] = clN;
        }
        
        for (uintT j = 0; j < clUNM.cells[i].connSize; j++) {
            CLConnection clC = toCLConnection(cell->netMod->nConns[j], cell->netSt->nCSt[j]);
            clUNM.cells[i].clCons[j] = clC;
        }
    }
    
    return clUNM;
}

UnifiedNeuralModel OCLBridge::fromCLUNM(const CLUnifiedNeuralModel &clUNM)
{
    UNMConfig conf = UNMConfig(INPUT_SIZE, OUTPUT_SIZE, CELL_POPULATION, STEP_MUTATION, 0, 0, NMSIZE);
    UnifiedNeuralModel unm = UnifiedNeuralModel(0, conf);
    
    for (uintT i = 0; i < NMSIZE; i++) {
        unm.nmap.nmStrs[i].weight = clUNM.nmap.nmStrs[i].weight;
        ushortT cellRef = clUNM.nmap.nmStrs[i].cellRef;
        if (cellRef != OUT_INDEX) {
            unm.nmap.nmStrs[i].ptr = (void *)unm.cells[cellRef];
        }
        else {
            unm.nmap.nmStrs[i].ptr = NULL;
        }
    }
    
    unm.nmap.lastIdx.ciDist = clUNM.nmap.lastIdx.ciDist;
    unm.nmap.lastIdx.ciIndex = clUNM.nmap.lastIdx.ciIndex;
    
    for (uintT i = 0; i < CELL_POPULATION; i++) {
        ushortT nrsSize = clUNM.cells[i].nrsSize;
        ushortT conSize = clUNM.cells[i].connSize;
        unm.cells[i]->cellFitness = clUNM.cells[i].clFitness;
        unm.cells[i]->cellStep = clUNM.cells[i].cellStep;
        unm.cells[i]->netMod->nParams.networkID = i;
        unm.cells[i]->netMod->nNeurons.resize(nrsSize);
        unm.cells[i]->netMod->nConns.resize(conSize);
        unm.cells[i]->netSt->nNSt.resize(nrsSize);
        unm.cells[i]->netSt->nCSt.resize(conSize);
        
        for (uintT j = 0; j < nrsSize; j++) {
            auto result = fromCLNeuron(clUNM.cells[i].clNrs[j]);
            unm.cells[i]->netMod->nNeurons[j].nID = result.first.nID;
            unm.cells[i]->netMod->nNeurons[j].firingRate = result.first.firingRate;
            unm.cells[i]->netMod->nNeurons[j].nType = result.first.nType;
            unm.cells[i]->netSt->nNSt[j].excitation = result.second.excitation;
            unm.cells[i]->netSt->nNSt[j].isFired = result.second.isFired;
            unm.cells[i]->netSt->nNSt[j].prevState = result.second.prevState;
            unm.cells[i]->netSt->nNSt[j].state = result.second.state;
        }
        
        for (uintT j = 0; j < conSize; j++) {
            auto res = fromCLConnection(j, clUNM.cells[i].clCons[j]);
            unm.cells[i]->netMod->nConns[j].fromNID = res.first.fromNID;
            unm.cells[i]->netMod->nConns[j].toNID = res.first.toNID;
            unm.cells[i]->netMod->nConns[j].neuroMod = res.first.neuroMod;
            unm.cells[i]->netMod->nConns[j].weight = res.first.weight;
            unm.cells[i]->netSt->nCSt[j].connID = res.second.connID;
            unm.cells[i]->netSt->nCSt[j].connType = res.second.connType;
            unm.cells[i]->netSt->nCSt[j].prevConnType = res.second.prevConnType;
        }
        
        unm.cells[i]->netSt->module = unm.cells[i]->netMod;
    }
    
    return unm;
}

void OCLBridge::checkConstantsMatch(const UnifiedNeuralModel &unm)
{
    // This function obtains the variable name and transforms it into a string.
    #define VAR_NAME(name) (#name)
    
    auto prtErr = [](string middle) {
        printf("¡¡NOTE!!\n");
        printf("%s\n", middle.c_str());
        printf("possible errors could be encountered when executing OpenCL kernels.\n");
        printf("This requires program recompilation to update the constant\n");
    };
    
    auto noteFtn = [](string fst, string snd, ushortT fstVal, ushort sndVal) {
        string note = fst + " != " + snd + " (";
        note += to_string(fstVal);
        note += " != ";
        note += to_string(sndVal) + ")";
        return note;
    };
    
    auto testVars = [&noteFtn, &prtErr](string fst, string snd, ushortT fstVal, ushort sndVal) {
        if (fstVal != sndVal) {
            string note = noteFtn(fst, snd, fstVal, sndVal);
            prtErr(note);
        }
    };
    
    ushortT episodesPerAgent = PConfig::globalProgramConfiguration()->episodesPerAgent();
    testVars(VAR_NAME(unm.config.unmPopulation), VAR_NAME(CELL_POPULATION), unm.config.unmPopulation, CELL_POPULATION);
    testVars(VAR_NAME(unm.config.unmMapSize), VAR_NAME(NMSIZE), unm.config.unmMapSize, NMSIZE);
    testVars(VAR_NAME(unm.config.unmObj), VAR_NAME(INPUT_SIZE), unm.config.unmObs, INPUT_SIZE);
    testVars(VAR_NAME(unm.config.unmActions), VAR_NAME(OUTPUT_SIZE), unm.config.unmActions, OUTPUT_SIZE);
    testVars(VAR_NAME(episodesPerAgent), VAR_NAME(EPISODES_PER_AGENT), episodesPerAgent, EPISODES_PER_AGENT);
    
}
