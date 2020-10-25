//
//  NNetworkExtra.cpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 27/6/19.
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

#include "NNetworkExtra.hpp"
#include "Parameters.hpp"
#include <cstdio>
#include <glob.h>
#include <sstream>
#include <iostream>
#include <fstream>

void NNExFunction::checkRemove(const char *fName)
{
    ifstream f(fName);
    
    if (f.good()) {
        remove(fName);
    }
    
    f.close();
}

bool NNExFunction::checkFile(const char *fName)
{
    ifstream f(fName);
    bool val = f.good();
    f.close();
    return val;
}

void NNExFunction::writeNetworkModule(const NNetworkModule *mod, ofstream &fileStrm)
{
    ushortT netParamSize = sizeof(NNetworkParams);
    ushortT nrsStructSize = sizeof(NNeuron);
    ushortT conStructSize = sizeof(NConnection);
    ushortT ushortSize = sizeof(ushortT);
    ushortT neuronsSize = mod->nNeurons.size();
    ushortT connSize = mod->nConns.size();
    
    fileStrm.write((char *)&(mod->nParams), netParamSize);
    fileStrm.write((char *)&neuronsSize, ushortSize);
    fileStrm.write((char *)mod->nNeurons.data(), nrsStructSize * neuronsSize);
    fileStrm.write((char *)&connSize, ushortSize);
    fileStrm.write((char *)mod->nConns.data(), conStructSize * connSize);
}

void NNExFunction::readNetworkModule(ifstream &fileStrm, NNetworkModule *mod)
{
    ushortT netParamSize = sizeof(NNetworkParams);
    ushortT nrsStructSize = sizeof(NNeuron);
    ushortT conStructSize = sizeof(NConnection);
    ushortT ushortSize = sizeof(ushortT);
    ushortT neuronsSize = 0;
    ushortT connSize = 0;
    
    fileStrm.read((char *)&(mod->nParams), netParamSize);
    fileStrm.read((char *)&neuronsSize, ushortSize);
    vector<NNeuron> vecNrs(neuronsSize);
    fileStrm.read((char *)vecNrs.data(), nrsStructSize * neuronsSize);
    fileStrm.read((char *)&connSize, ushortSize);
    vector<NConnection> vecConns(connSize);
    fileStrm.read((char *)vecConns.data(), conStructSize * connSize);
    
    mod->nNeurons = vecNrs;
    mod->nConns = vecConns;
}

NNetworkModule NNExFunction::loadNetworkModule(const char *filename)
{
    NNetworkModule mod = NNetworkModule();
    ifstream rf(filename, ios::out | ios::binary);
    readNetworkModule(rf, &mod);
    rf.close();
    return mod;
}

bool NNExFunction::saveNetworkModule(const NNetworkModule &mod, const char *filename)
{
    ofstream wf(filename, ios::out | ios::binary);
    writeNetworkModule(&mod, wf);
    wf.close();
    
    return wf.good();
}

void NNExFunction::writeNetworkState(const NNetworkState *nSt, ofstream &fileStrm)
{
    ushortT ushortSize = sizeof(ushortT);
    ushortT nsStructSize = sizeof(NNeuronState);
    ushortT csStructSize = sizeof(NConnState);
    ushortT ioElemSize = sizeof(*(nSt->ioVal.begin()));
    ushortT nstSize = nSt->nNSt.size();
    ushortT cstSize = nSt->nCSt.size();
    ushortT iotSize = nSt->ioVal.size();
    
    // In order to preserve the correspondance State-Module, it needs to first
    // write the corresponding module, then all state values
    writeNetworkModule(nSt->module, fileStrm);
    
    fileStrm.write((char *)&nstSize, ushortSize);
    fileStrm.write((char *)nSt->nNSt.data(), nsStructSize * nstSize);
    fileStrm.write((char *)&cstSize, ushortSize);
    fileStrm.write((char *)nSt->nCSt.data(), csStructSize * cstSize);
    fileStrm.write((char *)&iotSize, ushortSize);
    
    for (auto elem : nSt->ioVal) {
        fileStrm.write((char *)&elem, ioElemSize);
    }
}

void NNExFunction::readNetworkState(ifstream &fileStrm, NNetworkState *nSt)
{
    ushortT ushortSize = sizeof(ushortT);
    ushortT nsStructSize = sizeof(NNeuronState);
    ushortT csStructSize = sizeof(NConnState);
    ushortT ioElemSize = sizeof(pair<ushortT, ParameterType>);
    ushortT nstSize = 0;
    ushortT cstSize = 0;
    ushortT iotSize = 0;
    
    // In order to preserve the correspondance State-Module, it needs to first
    // write the corresponding module, then all state values
    readNetworkModule(fileStrm, nSt->module);
    
    fileStrm.read((char *)&nstSize, ushortSize);
    vector<NNeuronState>nst(nstSize);
    fileStrm.read((char *)nst.data(), nsStructSize * nstSize);
    fileStrm.read((char *)&cstSize, ushortSize);
    vector<NConnState>cst(cstSize);
    fileStrm.read((char *)cst.data(), csStructSize * cstSize);
    fileStrm.read((char *)&iotSize, ushortSize);
    unordered_map<ushortT, ParameterType> ioMap;
    
    for (ushort i = 0; i < iotSize; i++) {
        pair<ushortT, ParameterType> elem;
        fileStrm.read((char *)&elem, ioElemSize);
        ioMap.insert(elem);
    }
    
    nSt->nNSt = nst;
    nSt->nCSt = cst;
    nSt->ioVal = ioMap;
}

void NNExFunction::writeUNMCell(const UNMCell *cell, ofstream &fileStrm)
{
    writeNetworkState(cell->netSt, fileStrm);
    
    fileStrm.write((char *)&(cell->cellFitness), sizeof(float));
    fileStrm.write((char *)&(cell->cellStep), sizeof(uintT));
}

void NNExFunction::readUNMCell(ifstream &fileStrm, UNMCell *cell)
{
    readNetworkState(fileStrm, cell->netSt);
    
    fileStrm.read((char *)&(cell->cellFitness), sizeof(float));
    fileStrm.read((char *)&(cell->cellStep), sizeof(uintT));
}

void NNExFunction::writeNoveltyMap(const NNoveltyMap *nmap, ofstream &fileStrm)
{
    ushortT ushortSize = sizeof(ushortT);
    ushortT strIndexSize = sizeof(NMStrIndex);
    ushortT ulongSize = sizeof(ulongT);
    ushortT strSize = nmap->nmStrs.size();
    
    fileStrm.write((char *)&(nmap->popSize), ushortSize);
    fileStrm.write((char *)&(nmap->lastIdx), strIndexSize);
    fileStrm.write((char *)&strSize, ushortSize);
    
    for (NMStr str : nmap->nmStrs) {
        fileStrm.write((char *)&(str.weight), ulongSize);
    }
}

void NNExFunction::readNoveltyMap(ifstream &fileStrm, NNoveltyMap *nmap)
{
    ushortT ushortSize = sizeof(ushortT);
    ushortT strIndexSize = sizeof(NMStrIndex);
    ushortT ulongSize = sizeof(ulongT);
    ushortT strSize = 0;
    
    fileStrm.read((char *)&(nmap->popSize), ushortSize);
    fileStrm.read((char *)&(nmap->lastIdx), strIndexSize);
    fileStrm.read((char *)&strSize, ushortSize);
    
    vector<NMStr> strs;
    for (ushortT i = 0; i < strSize; i++) {
        ulongT weight = 0;
        fileStrm.read((char *)&(weight), ulongSize);
        strs.push_back(NMStr(weight));
    }
    
    nmap->nmStrs = strs;
}

void NNExFunction::writeUNM(const UnifiedNeuralModel *model, ofstream &fileStrm)
{
    ushortT cellSize = model->cells.size();
    
    fileStrm.write((char *)&(model->config), sizeof(UNMConfig));
    
    writeNoveltyMap(&(model->nmap), fileStrm);
    
    fileStrm.write((char *)&cellSize, sizeof(ushortT));
    
    for (UNMCell *cell : model->cells) {
        writeUNMCell(cell, fileStrm);
    }
}

void NNExFunction::readUNM(ifstream &fileStrm, UnifiedNeuralModel *model)
{
    ushortT cellSize = 0;
    
    fileStrm.read((char *)&(model->config), sizeof(UNMConfig));
    
    readNoveltyMap(fileStrm, &(model->nmap));
    
    fileStrm.read((char *)&cellSize, sizeof(ushortT));
    
    vector<UNMCell *>cells;
    for (ushortT i = 0; i < cellSize; i++) {
        NNetworkModule *mod = new NNetworkModule();
        NNetworkState *st = new NNetworkState(mod);
        UNMCell *cell = new UNMCell(mod, st);
        cell->deallocate = true;
        readUNMCell(fileStrm, cell);
        cells.push_back(cell);
    }
    
    model->cells = cells;
}

bool NNExFunction::writeBiSUNAModel(const UnifiedNeuralModel *model, const char *fileName)
{
    checkRemove(fileName);
    ofstream wf(fileName, ios::out | ios::binary);
    writeUNM(model, wf);
    wf.close();
    return wf.good();
}

void NNExFunction::readBiSUNAModel(const char *fileName, UnifiedNeuralModel *model)
{
    ifstream rf(fileName, ios::in | ios::binary);
    
    if (rf.good()) {
        readUNM(rf, model);
        rf.close();
    }
    else {
        printf("File %s could not be read correctly, finishing execution", fileName);
        rf.close();
        exit(1);
    }
}

string NNExFunction::appendTimeStamp(const string &name)
{
    uintT timeNow = static_cast<unsigned int>(time(NULL));
    string strTime = to_string(timeNow);
    return appendSuffixString(name, strTime);
}

string NNExFunction::appendSuffixString(const string &name, const string &suffix)
{
    auto idx = name.rfind('.');
    string update = name;
    update.insert(idx, suffix);
    return update;
}

vector<string> NNExFunction::allFilesInFolder(string folder)
{
    vector<string> result;
    folder.append("/*");
    glob_t glob_result;
    glob(folder.c_str(), GLOB_TILDE, NULL, &glob_result);
    for(ushortT i = 0; i < glob_result.gl_pathc; i++) {
        result.push_back(glob_result.gl_pathv[i]);
    }
    
    return result;
}

void NNExFunction::removeOrphans(UNMCell *cell)
{
    set<ushortT> nIDs;
    
    for (NConnection c : cell->netMod->nConns) {
        if (c.fromNID == c.toNID) {
            continue; // Self connections do not count.
        }
        
        nIDs.insert(c.fromNID);
        nIDs.insert(c.toNID);
        if (c.neuroMod != OUT_INDEX) {
            nIDs.insert(c.neuroMod);
        }
    }
    
    ushort nrsSize = cell->netMod->nNeurons.size();
    ushort ioNrs = cell->netMod->nParams.nInputs + cell->netMod->nParams.nOutputs;
    ushort offset = 0;
    vector<ushortT> removedNeurons;
    
    if (nIDs.size() == nrsSize) {
        return; // There is no need to check orphans because all of them have at
        // least one in/out connection
    }
    
    for (ushortT i = ioNrs; i < nrsSize; i++) {
        NNeuron n = cell->netMod->nNeurons[i];
        ushort offNID = n.nID + offset;
        auto pos = nIDs.find(offNID);
        
        if (pos == nIDs.end()) {
            NNeuron removed = NNMFunction::deleteNeuron(n.nID, cell->netMod->nNeurons);
            NNMFunction::deleteNeuronConnections(n.nID, cell->netMod->nConns);
            removedNeurons.push_back(removed.nID);
            offset++;
            i--;
            nrsSize--;
        }
    }
    
    cell->netSt->nNSt.resize(cell->netMod->nNeurons.size());
    cell->netSt->nCSt.resize(cell->netMod->nConns.size());
}

void NNExFunction::printGraph(const char *filename, NNetworkModule *module)
{
    ushort inputs = module->nParams.nInputs;
    ushort outputs = module->nParams.nOutputs;
    vector<NNeuron> neurons = module->nNeurons;
    vector<NConnection> conns = module->nConns;
    size_t nSize = neurons.size();
    size_t cSize = conns.size();
    string sstring = "";
    stringstream ss;
    ss << "digraph G {\nnode[shape = circle]" << endl;
    ss << "rankInputs [style = invisible]\nrankOutputs [style = invisible]" << endl;
    ss << "rankInputs -> rankOutputs [color=invis]" << endl;
        
    for (uintT i = 0; i < nSize; i++)
    {
        NNeuron n = neurons[i];

        switch(n.nType)
        {
//ntID, ntActivation, ntThreshold, ntRandom, ntControl, ntCPrimer, ntInput, ntOutput
            case ntID: {
                sstring = "style = filled, fillcolor = gold";
                break;
            }
            case ntActivation: {
                sstring = "style = filled, fillcolor = lightskyblue";
                break;
            }
            case ntThreshold: {
                sstring = "style = filled, fillcolor=limegreen";
                break;
            }
            case ntRandom: {
                sstring = "style = filled, fillcolor = coral";
                break;
            }
            case ntControl: {
                sstring = "shape = octagon";
                break;
            }
            case ntCPrimer: {
                sstring = "shape = doubleoctagon";
                break;
            }
            case ntRoll: {
                sstring = "shape = trapezium";
                break;
            }
            case ntInput: {
                sstring = "style = filled, shape = invhouse, color=lightblue";
                break;
            }
            case ntOutput: {
                sstring = "style = filled, shape = house, color=tan";
                break;
            }
            default: {
                printf("No neuron should be in this part. Finishing execution\n");
                exit(1);
            }
            break;
        }

        ss << i << " [label=\"#" << i << ":" << (ushortT)n.firingRate;
        ss << "\", " << sstring << "]" << endl;
        sstring = "";
    }

    size_t counter = 1;
    
    for(uintT i = 0; i < cSize; i++) {
        NConnection c = conns[i];

        if(c.neuroMod == OUT_INDEX) {//no neuro modulation
            ss << c.fromNID << " -> " << c.toNID << " [label = \"" << c.weight << "\"]" << endl;
        }
        else { // With NeuroModulation
            uintT nModPoint = (uintT)(nSize + counter);
            ss << nModPoint << " [shape = point]" << endl;
            ss << c.neuroMod << " -> " << nModPoint << " [style = dashed]" << endl;
            ss << c.fromNID << " -> " << nModPoint << " [dir = none]" << endl;
            ss << nModPoint << " -> " << c.toNID << endl;
            counter++;
        }
    }
    
    auto addRank = [](string name, vector<ushortT> ids) {
        stringstream ss;
        ss << "{\n\trank = same\n\trankdir = TB\n\trank" << name;
        for (ushortT i : ids) {
            ss << " -> " << i;
        }
        ss << " [style = invis]" << endl;
        ss << "}" << endl;
        return ss.str();
    };
    
    auto vecSection = [](size_t first, size_t last, vector<NNeuron> vec) {
        vector<ushortT> res;
        for (size_t i = first; i < last; i++) {
            res.push_back(vec[i].nID);
        }
        return res;
    };

    ss << addRank("Inputs", vecSection(0, inputs, module->nNeurons));
    ss << addRank("Outputs", vecSection(inputs, inputs + outputs, module->nNeurons));
    ss << "}" << endl;
    
    ofstream fp;
    fp.open(filename, ios::out | ios::binary);
    fp << ss.str();
    fp.close();
}

bool NNExFunction::writeToBinFile(const string &filePath, const ParameterVector &payload)
{
    ofstream wf(filePath, ios::out | ios::binary);
    size_t dataSize = payload.size() * sizeof(ParameterType);
    char *ptr = (char *)(payload.data());
    wf.write(ptr, dataSize);
    wf.close();
    return wf.good();
}

ParameterVector NNExFunction::readFromBinFile(const string &filePath)
{
    ifstream rf(filePath, ios::in | ios::binary);
    rf.seekg(0, std::ios_base::end);
    size_t dataSize = rf.tellg();
    rf.seekg(0, std::ios_base::beg);

    size_t numElems = dataSize / sizeof(ParameterType);
    ParameterVector elems(numElems);
    
    char *ptr = (char *)elems.data();
    if (rf.good()) {
        rf.read(ptr, dataSize);
        rf.close();
    }
    else {
        printf("File %s could not be read correctly, finishing execution", filePath.c_str());
        rf.close();
        exit(1);
    }
    
    return elems;
}

