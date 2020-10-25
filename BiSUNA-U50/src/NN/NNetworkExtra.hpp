//
//  NNetworkExtra.hpp
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

#ifndef NNetworkExtra_hpp
#define NNetworkExtra_hpp

#include <fstream>
#include "RLAgent/UnifiedNeuralModel.hpp"

struct NNExFunction
{
    static void checkRemove(const char *fName);
    static bool checkFile(const char *fName);
    static void writeNetworkModule(const NNetworkModule *mod, ofstream &fileStrm);
    static void readNetworkModule(ifstream &fileStrm, NNetworkModule *mod);
    static NNetworkModule loadNetworkModule(const char *filename);
    static bool saveNetworkModule(const NNetworkModule &mod, const char *filename);
    static void writeNetworkState(const NNetworkState *nSt, ofstream &fileStrm);
    static void readNetworkState(ifstream &fileStrm, NNetworkState *nSt);
//    static void write(const  *cell, ofstream &fileStrm);
//    static void read(ifstream &fileStrm,  *cell);
    static void writeUNMCell(const UNMCell *cell, ofstream &fileStrm);
    static void readUNMCell(ifstream &fileStrm, UNMCell *cell);
    static void writeNoveltyMap(const NNoveltyMap *nmap, ofstream &fileStrm);
    static void readNoveltyMap(ifstream &fileStrm, NNoveltyMap *nmap);
    static void writeUNM(const UnifiedNeuralModel *model, ofstream &fileStrm);
    static void readUNM(ifstream &fileStrm, UnifiedNeuralModel *model);
    static bool writeBiSUNAModel(const UnifiedNeuralModel *model, const char *fileName);
    static void readBiSUNAModel(const char *fileName, UnifiedNeuralModel *model);
    static string appendTimeStamp(const string &name);
    static string appendSuffixString(const string &name, const string &suffix);
    static vector<string> allFilesInFolder(string folder);
    static void removeOrphans(UNMCell *cell);
    static void printGraph(const char *filename, NNetworkModule *module);
    static bool writeToBinFile(const string &filePath, const ParameterVector &payload);
    static ParameterVector readFromBinFile(const string &filePath);
};
#endif /* NNetworkExtra_hpp */
