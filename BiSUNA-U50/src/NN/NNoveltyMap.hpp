//
//  NNoveltyMap.hpp
//  BiSUNAOpenCL
//
//  Created by RHVT on 3/6/19.
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

#ifndef NNoveltyMap_hpp
#define NNoveltyMap_hpp

#include "Configuration/PConfig.hpp"

// Novelty Map Store
struct NMStr
{
    ulongT weight;
    void *ptr; // In order to keep values referenced in this cell
    
    NMStr(const ulongT value = 0, void *pointer = NULL):
        weight(value), ptr(pointer) {};
};

struct NMStrIndex
{
    ushortT ciDist;
    ushortT ciIndex;
    
    NMStrIndex(const ushortT dist = OUT_INDEX,
               const ushortT idx = 0):
        ciDist(dist), ciIndex(idx) {};
};

struct NNoveltyMap
{
    // When creating a novelty map, it should have the initial values that are going to
    // be stored in the nmap, that will set the population size and will calculate the
    // minimum distance between individuals
    NNoveltyMap(const vector<ulongT> &input = {});
    ~NNoveltyMap();
    
    vector<NMStr> nmStrs;
    ushortT popSize;
    NMStrIndex lastIdx;
};

struct NNMapFunction
{
    static ushortT distanceBetween(const ulongT a, const ulongT b);
    static NMStrIndex minDistance(const ulongT input, const vector<NMStr> &nmStrs, const int idx = -1);
    static NMStrIndex vectorMinDistance(const vector<NMStr> &nmStrs);
    static void replaceValue(const ulongT input, NNoveltyMap *nmap);
    static ushortT idxPosition(const ulongT input, NNoveltyMap *nmap);
};

#endif /* NNoveltyMap_hpp */
