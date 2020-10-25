//
//  NNoveltyMap.cpp
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

#include "NNoveltyMap.hpp"
#include "assert.h"

NNoveltyMap::NNoveltyMap(const vector<ulongT> &input)
{
    popSize = input.size();
    
    for (ulongT value : input) {
        nmStrs.push_back(NMStr(value));
    };
    
    lastIdx = NNMapFunction::vectorMinDistance(nmStrs);
}

NNoveltyMap::~NNoveltyMap()
{
}

/**
 A simple popcount operation over two 64 bits variables. An example is:
 a = 0x2C (0010 1100)
 b = 0x7C (0111 1100)
 c = 2
 The example above considers only a byte, but this function will work on
 64 bits operants
 @param a First operant
 @param b Second operant
 @return Hamming distance between two elements
 */
inline ushortT NNMapFunction::distanceBetween(const ulongT a, const ulongT b)
{
    ulongT c = a ^ b;
    return __builtin_popcountll(c);
}

/**
  Calculate the minimal distance comparing the input to the nmStrs vector, which
  creates an NMStrIndex with the minimum distance to the input along the index to
  the closest individual.

 @param input Value that will be compared against the vector
 @param nmStrs Cells with all the elements in the novelty map
 @param idx Let the function know if the input is already in nmStrs signaling its
 index, if that happens, the search will not compare an element at that index
 @return NMStrIndex with the minimum distance and the index most similar to the
 input provided
 */
NMStrIndex NNMapFunction::minDistance(const ulongT input, const vector<NMStr> &nmStrs, const int idx)
{
    size_t nmSize = nmStrs.size();
    assert(nmSize > 0);
    
    NMStrIndex elem = NMStrIndex();
    
    for (ushortT i = 0; i < nmSize; i++) {
        ulongT w = nmStrs[i].weight;
        
        if (idx == (int)i) {
            continue;
        }
        
        if (input == w) {
            elem.ciDist = 0;
            elem.ciIndex = i;
            break;
        }
        
        ushortT currDist = distanceBetween(w, input);
        if (currDist < elem.ciDist) {
            elem.ciDist = currDist;
            elem.ciIndex = i;
        }
    }
    
    return elem;
}

/**
  Given a NMStr vector, it will return the minimum distance that all components
  in the nmStr are currently against each other.

 @param nmStrs nm store vector that contains all elements a the novelty map
 @return Updated NMStrIndex that calculates the minimum distance and the index
 element which has that least distance.
 */
NMStrIndex NNMapFunction::vectorMinDistance(const vector<NMStr> &nmStrs)
{
    ushortT strSize = nmStrs.size();
    NMStrIndex idx = NMStrIndex();
    if (strSize <= 1) {
        return idx;
    }
    
    for (ushortT i = 0; i < strSize; i++) {
        NMStrIndex minIdx = minDistance(nmStrs[i].weight, nmStrs, i);
        if (idx.ciDist > minIdx.ciDist) {
            idx.ciDist = minIdx.ciDist;
            idx.ciIndex = i;
        }
    }
        
    return idx;
}

/**
 This function will replace the "worst individual" with the value
 provided as "input" into the novelty map. Also the "lastIdx" property
 will be updated.

 @param input The value that will be attached to the nmap
 @param nmap Current novelty map that is used to test and modify
 */
void NNMapFunction::replaceValue(const ulongT input, NNoveltyMap *nmap)
{
    // Any new elements will replace the "worst individual" given
    // a previously calculated distance of lesser value than what it
    // is intended to append.
    ushortT worstInd = nmap->lastIdx.ciIndex;
    nmap->nmStrs[worstInd].weight = input;
    nmap->lastIdx = vectorMinDistance(nmap->nmStrs);
};

/**
  Return the index position int the nmap closest to the input, this way it is
  possible to compare if the index to closest cell has stored a value or not.
 
 @param input The value that must be compared to other elements in the nmap
 @param nmap Current novelty map that is used to test and modify
 @return index which represents the location in the nmap. This element is
    obtained by returning the individual with least distance stored ptr OR
    the closest relative to the input provided
 */
ushortT NNMapFunction::idxPosition(const ulongT input, NNoveltyMap *nmap)
{
    auto res = minDistance(input, nmap->nmStrs);
    bool lessDist = nmap->lastIdx.ciDist < res.ciDist;
    if (lessDist) {
        replaceValue(input, nmap);
        return nmap->lastIdx.ciIndex;
    }
    
    return res.ciIndex;
}
