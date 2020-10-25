//
//  CLNetState.cl
//  BiSUNAOpenCLTests
//
//  Created by RHVT on 19/6/20.
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


//#define CONTINUOUS_PARAM

#ifdef CONTINUOUS_PARAM
    typedef float CLPType;
    #define EXCITATION_THRESHOLD 0.0    //minimum excitation necessary to activate the neuron
    #define REMAINING_NEURON_THRESHOLD 0.001
    #define WEIGHT_MUTATION_CHANGE_PERCENTAGE 1.0 //10 = 1000% change, 1 = 100% change possible
    #define EXCITATION_THRESHOLD_BITS 4 // Used only in the binary part, but redefined here for compilation purposes
    #define HALF_WEIGHT 0
    #define MID_WEIGHT 0
    #ifndef MAXIMUM_WEIGHT
        #define MAXIMUM_WEIGHT 2147483647.0
    #endif
#else
    typedef ushort CLPType;
    #define EXCITATION_THRESHOLD 256//minimum excitation necessary to deactivate the neuron
    // This constant considers the number of bits an excitation neuron must have in order
    // to trigger its actions.
    #define REMAINING_NEURON_THRESHOLD 15
    #define WEIGHT_MUTATION_CHANGE_PERCENTAGE 1
    #define EXCITATION_THRESHOLD_BITS 4
//    #define EXCITATION_THRESHOLD_BITS 6
// This represents the half part of all bits in a ushort (16 bits) set to one, it is used by primer, control
// and threshold neurons to check its trigger action
    #define HALF_WEIGHT 255
// Represents the actual mid section of the ushort type, use only in the discretization functions to transform
// between continuous inputs/outputs
    #define MID_WEIGHT 32767
    #ifndef MAXIMUM_WEIGHT
        #define MAXIMUM_WEIGHT USHRT_MAX // 65535
    #endif
#endif

#ifndef INITIALBATCH
#define INITIALBATCH 1000
#endif

#ifndef INITIALBATCHX2
#define INITIALBATCHX2 (INITIALBATCH * 2)
#endif

#ifndef NMSIZE
#define NMSIZE 20
#endif

#ifndef CELL_POPULATION
#define CELL_POPULATION 100
#endif

#ifndef INPUT_SIZE
#define INPUT_SIZE 2
#endif

#ifndef OUTPUT_SIZE
#define OUTPUT_SIZE 3
#endif

#ifndef IO_SIZE
#define IO_SIZE (INPUT_SIZE + OUTPUT_SIZE)
#endif

#define ARRAY_MAX 128
#define MAX_FIRING_RATE 3
#define MAX_NEURON_TYPE 5

#ifndef OUT_INDEX
#define OUT_INDEX USHRT_MAX // 65535
#endif

// PConfig default values
#ifndef CHANCE_OF_CONTROL_NEURON
#define CHANCE_OF_CONTROL_NEURON 20 // 20% -> 0.2
#endif

#ifndef CHANCE_OF_NEUROMODULATION
#define CHANCE_OF_NEUROMODULATION 10 // 10% -> 0.1
#endif

#ifndef CHANCE_OF_ADD_NEURON
#define CHANCE_OF_ADD_NEURON 2 // 1% -> 0.01
#endif

#ifndef CHANCE_OF_DEL_NEURON
#define CHANCE_OF_DEL_NEURON 2 // 1% -> 0.01
#endif

#ifndef CHANCE_OF_ADD_CONN
#define CHANCE_OF_ADD_CONN 48 // 49% -> 0.49
#endif

#ifndef CHANCE_OF_DEL_CONN
#define CHANCE_OF_DEL_CONN 48 // 49% -> 0.49
#endif

#ifndef CHANCE_OF_WEIGHT_MUT
#define CHANCE_OF_WEIGHT_MUT 50  // weightMutProb 50% -> 0.5
#endif

#ifndef STEP_MUTATION
#define STEP_MUTATION 5
#endif

#ifndef EPISODES_PER_AGENT
#define EPISODES_PER_AGENT 1
#endif

#define PIPELINE

typedef struct __attribute__((packed)) __attribute__((aligned(4)))
{
    CLPType clWeight;
    ushort clFromNID;
    ushort clToNID;
    ushort clNeuroMod;
    
    // Part of the CLConnState
    bool clCnType; // NOTE: false = ctRecurrent, true = ctForward
    bool clPCnType; // NOTE: false = ctRecurrent, true = ctForward
    // Part of the CLConnState
} CLConnection;


typedef struct __attribute__((packed)) __attribute__((aligned(4)))
{
    // Part of CLNeuronState
    CLPType clSt;
    CLPType clPSt;
    CLPType clExc;
    // Part of CLNeuronState
    
    // Part of CLNeuron
    ushort clNID;
    
    // clFiringRate: index 1 (6 most significant bits 1111111 0000000000)
    // clNType: index 1 (10 least significant bits 000000 1111111111)
    ushort clFRNT;
    // Part of CLNeuron
    
    bool clFired;
} CLNeuron;


typedef struct __attribute__((packed)) __attribute__((aligned(4)))
{
    CLConnection clCons[INITIALBATCHX2];
    CLNeuron clNrs[INITIALBATCH];
    float clFitness; // TODO!! check later on for a full BNN execution
    uint cellStep;
    ushort clNetworkID;
    ushort nrsSize;
    ushort connSize;
} CLCell;


typedef struct __attribute__((packed)) __attribute__((aligned(4)))
{
    ulong weight;
    ushort cellRef; // In order to keep values referenced in this cell
} CLNMStr;


typedef struct __attribute__((packed)) __attribute__((aligned(4)))
{
    ushort ciDist;
    ushort ciIndex;
} CLNMStrIndex;


typedef struct __attribute__((packed)) __attribute__((aligned(4)))
{
    CLNMStr nmStrs[NMSIZE];
    CLNMStrIndex lastIdx;
} CLNoveltyMap;


typedef struct __attribute__((packed)) // __attribute__((aligned(4)))
{
    CLCell cells[CELL_POPULATION];
    CLNoveltyMap nmap;
    uint clGeneration;
    uint clSteps;
} CLUnifiedNeuralModel;


//void printCell(global CLCell *cell, ushort size)
//{
//	printf("---////////////---\n");
//	for (ushort i = 0; i < size; i++) {
//		global CLNeuron *nrs = cell[i].clNrs;
//		global CLConnection *cons = cell[i].clCons;
//		ushort nrsSize = cell[i].nrsSize;
//		ushort connSize = cell[i].connSize;
//		printf("CLCell %i, Nrs %i [\n", cell[i].clNetworkID, nrsSize);
//		CLNeuron nrn = nrs[0];
//		CLConnection conn = cons[0];
//		for (ushort j = 0; j < nrsSize; j++) {
//			printf("%i\t%i\t%i\t%i\t%i\t%i\n", nrn.clNID, nrn.clFRNT, nrn.clSt, nrn.clPSt, nrn.clExc, nrn.clFired);
//			nrn = nrs[j+1];
//		}
//		printf("]\nCons %i [\n", connSize);
//		for (ushort j = 0; j < connSize; j++) {
//			printf("%i\t%i\t%i\t%i\t%i\n", conn.clFromNID, conn.clToNID, conn.clNeuroMod, conn.clCnType, conn.clPCnType);
//			conn = cons[j+1];
//		}
//		printf("]===\\\\\\\\\\\\===\n");
//	}
//}


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

inline CLPType activationFunction(ushort nt, CLPType weightedInput)
{
    // Mask only the first 10 bits
    nt = nt & 2047;
    switch(nt)
    {
        case 128: { // ntRandom:
#ifdef CONTINUOUS_PARAM
//            float val = tycheIFloat((*tSt));
            float val = 0;
#else
//            ushort val = tycheIUShort((*tSt));
            ushort val = 0;
#endif
            return val;
        }
        case 32: { // ntActivation
#ifdef CONTINUOUS_PARAM
            return tanh(weightedInput);
#else
            return popcount(weightedInput);
#endif
        }
        case 512: { // ntRoll
#ifdef CONTINUOUS_PARAM
            float change = weightedInput >= 0 ?  1.0 : -1.0;
            return weightedInput + change;
#else
            ushort val = weightedInput << 1 | weightedInput >> 15;
            return val;
#endif
        }
        case 1024:
        case 64: // ntThreshold
        case 256: { // ntControl
#ifdef CONTINUOUS_PARAM
            return weightedInput >= 0 ? 1.0 : -1.0;
#else
            int val = popcount(weightedInput);
            return val >= EXCITATION_THRESHOLD_BITS ? HALF_WEIGHT : 0;
#endif
        }
        case 16: // ntID:
        case 1: // ntInput:
        case 2: { // ntOutput: {
            return weightedInput;
        }
        case 6: // ntNumberNeuronType:
        default: return 0;
    }
}

inline bool checkExcitationThreshold(CLPType ne)
{
#ifdef CONTINUOUS_PARAM
    return ne >= EXCITATION_THRESHOLD;
#else
    return ne < EXCITATION_THRESHOLD;
#endif
}

inline ushort clFiringRateByIndex(ushort idx)
{
    switch (idx) {
        case 0: return 1; // frL1
        case 1: return 7; // frL7
        case 2: return 49; // frL49
        default: return MAX_FIRING_RATE; // frNumberFiringRate;
    }

    return MAX_FIRING_RATE;
}

inline CLPType betaOfFiringRate(ushort fr)
{
    // We need to move all bits 10 places to remove Neuron type
    // information and read only the firing rate value.
    fr = fr >> 11;
    
#ifdef CONTINUOUS_PARAM
    float frD = convert_float(clFiringRateByIndex(fr));
    float partial = 1 / frD;
    return partial;
#else
    switch(fr) {
        case 0: return OUT_INDEX;
        case 1: return 2047;
        case 2: return 63;
        default: return 0;
    }
#endif
}

inline CLPType stateActivation(ushort nFR, CLPType prevSt, CLPType neuronOutput)
{
    CLPType beta = betaOfFiringRate(nFR);
    
#ifdef CONTINUOUS_PARAM
    //use Windrow Hoff for slower neurons
    CLPType state = prevSt + beta * (neuronOutput - prevSt);
#else
    CLPType state = (prevSt ^ (beta & (~(neuronOutput ^ prevSt))));
#endif
    
    return state;
}

inline bool remainingNeuronThreshold(CLPType output)
{
#ifdef CONTINUOUS_PARAM
    return output > REMAINING_NEURON_THRESHOLD || output < -REMAINING_NEURON_THRESHOLD;
#else
    return output >= REMAINING_NEURON_THRESHOLD;
#endif
}

inline CLPType neuronOpr(CLPType a, CLPType b, CLPType c)
{
#ifdef CONTINUOUS_PARAM
    return c + (a * b);
#else
    return c | (a & b);
#endif
    
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////



// idx represents the index value along the st/mod array that identifies that specific
// network module/state. If the neuron properties are required, it should reference
// using idx, ex: neuron = neurons[idx].
CLPType execute(ushort idx,
                bool ignoreIfOnlyRecurrent,
                const ushort2 netSt,
                global CLNeuron *neurons,
                global CLConnection *conns,
                const global CLPType *input,
                global CLPType *output)
{
    // If they are added to this array, it means they have a "FeedForward" Connection state
    ushort usedConns[ARRAY_MAX];
    int usedConnsCounter = -1;

    CLPType sum = 0;
    bool onlyRecurrentInputs = true;

    CLNeuron neuron = neurons[idx];

    //printf("ExeFor: %i\n", idx);
    //check the connections to this neuron
    //    for (ushort i = mod.connsStart; i < mod.connsEnd; i++) {
    for (ushort i = 0; i < netSt.s1; i++) {
        CLConnection conn = conns[i];

        //skip if it is not a connection to this Neuron
        if (conn.clToNID != idx) {
            continue;
        }

        //        ushort source = mod.neuronsStart + conn.clFromNID;
        ushort source = conn.clFromNID; // mod.neuronsStart

        //add this input multiplied by the weight of the connection
        //IGNORE if it comes from a Control Neuron
        ushort t = neurons[source].clFRNT;
        // ntControl = 256, ntCPrimerControl = 512,
        bool notControl = !(t & (256 | 1024));
        if (notControl) {
            CLNeuron sourceSt = neurons[source];
            ushort modulator = conn.clNeuroMod;
            //no neuro modulation
            if (modulator == OUT_INDEX) {
                CLPType cw = conn.clWeight;

                //check if they were already activated
                if (sourceSt.clFired == true) {
                    sum = neuronOpr(cw, sourceSt.clSt, sum);
                    onlyRecurrentInputs = false;
                    //marking the recurrent connection as used to be processed later on
                    //because we still do not know if this neuron will fire
                    usedConnsCounter++;
                    usedConns[usedConnsCounter] = i;
                }
                //recurrent connection
                //case it was not activated yet, use the state from the previous iteration
                else {
                    CLPType pn = sourceSt.clPSt;

                    if (conn.clPCnType == false) { //ctRecurrent
                        sum = neuronOpr(cw, pn, sum);

                    }
                }
            }
            //with neuron modulation
            else {
                ///////////////////////////////////////////////////////////////
                //
                //    Neuron Modulation
                //
                //
                //    ----(Modulator Neuron)----
                //                 |
                //    ----(Source Neuron)---------------(This Neuron)---------
                //
                ///////////////////////////////////////////////////////////////

                /////////////// modulator neuron /////////////
                CLPType modInput = 0;
                CLPType sourceInput = 0;
                CLNeuron neuroModulator = neurons[modulator];
                CLPType ne = neuroModulator.clExc;

                //check if it is excited
                if (checkExcitationThreshold(ne)) {
                    //modulator already activated
                    if (neuroModulator.clFired == true) {
                        modInput = neuroModulator.clSt;
                    }
                    //modulator not already activated, use the state from the previous iteration
                    else {
                        modInput = neuroModulator.clPSt;
                    }
                }

                /////////////// source neuron ////////////
                //source already activated
                if (sourceSt.clFired == true) {
                    sourceInput = sourceSt.clSt;

                    //marking the recurrent connection as used to be processed later on
                    //because we still do not know if this neuron will fire
                    usedConnsCounter++;
                    usedConns[usedConnsCounter] = i;
                }
                //recurrent connection
                //source not already activated, use the state from the previous iteration
                else { // ctRecurrent;
                    sourceInput = sourceSt.clPSt;
                }

                sum = neuronOpr(modInput, sourceInput, sum);
            }
        }
    }

    // If this neuron is an input, we need to add the value from
    // the current state to the sum given that later is replaced
    if (neuron.clFRNT & 1) { // nInput
        sum += input[idx];
    }

    //In the case that the ignore_if_only_recurrent flag is set,
    //the neuron does not activate if it only has recurrent input
    //this is useful for normal "remaining neurons" (neurons that are not control, not input and not output neurons)
    if (onlyRecurrentInputs == true && ignoreIfOnlyRecurrent == true) {
        return 0;
    }

    //mark all feedback connections
    for (; usedConnsCounter >= 0; usedConnsCounter--) {
        ushort cID = usedConns[usedConnsCounter];
        conns[cID].clCnType = 1;
    }

    //printf("Act: %i, FRNT: %i, Sum: %i,\n", idx, neuron.clFRNT, sum);
    CLPType neuronOutput = activationFunction(neuron.clFRNT, sum);
    //printf("PreState: %i, Out: %i, FRNT: %i\n", idx, neuronOutput, neuron.clFRNT);
//    get_global_id(0) == 1 ? printf("NeuronOut: %i, CLFRNT: %i, Sum: %i\n", neuronOutput, neuron.clFRNT, sum) : 0;
    neurons[idx].clSt = stateActivation(neuron.clFRNT, neurons[idx].clPSt, neuronOutput);
    //printf("Pos: %i, %i, %i\n", idx, neurons[idx].clSt, neurons[idx].clPSt);

    if (neuron.clFRNT & 2) { // nOutput
        //        ushort offset = (st.outputIdx + neuron.clNID - st.inputSize);
        ushort offset = (idx - INPUT_SIZE);
        //printf("Offset: %i, IN, %i,\n", idx, neurons[idx].clSt, neurons[idx].clPSt);
        output[offset] = neuronOutput;
        //printf("Output: %i, %i, %i,\n", idx, neurons[idx].clSt, neurons[idx].clPSt);
    }

    return neuronOutput;
}

void processInputNeurons(const ushort2 netSt,
                         global CLNeuron *neurons,
                         global CLConnection *conns,
                         const global CLPType *input,
                         global CLPType *output)
{
    for (ushort idx = 0; idx < INPUT_SIZE; idx++) {
        execute(idx, false, netSt, neurons, conns, input, output);
    }

    for (ushort idx = 0; idx < INPUT_SIZE; idx++) {
        neurons[idx].clFired = true;
    }
}

void processOutputNeurons(const ushort2 netSt,
                          global CLNeuron *neurons,
                          global CLConnection *conns,
                          const global CLPType *input,
                          global CLPType *output)
{
    for (ushort idx = INPUT_SIZE; idx < IO_SIZE; idx++) {
        if (neurons[idx].clFired == false) {
            execute(idx, false, netSt, neurons, conns, input, output);
            neurons[idx].clFired = true;
        }
    }
}

void runControlID(ushort idx,
                  ushort connSize,
                  global CLConnection *conns,
                  global CLNeuron *neuronsSt)
{
    CLPType ns = neuronsSt[idx].clSt;

    // This means no precision is necessary, that way
    // when compiling for the FPGA, it will reduce the
    // II value for this function.
#ifndef PRECISION
    // Composed of destination, modulator, clExc
    ushort3 destMod[ARRAY_MAX];
    int destModCounter = -1;

    // Check the connections from this neuron
    for (ushort i = 0; i < connSize; i++) {
        CLConnection conn = conns[i];
        // Skip if it is not a connection from this Neuron
        if (conn.clFromNID != idx) {
            continue;
        }

        ushort3 dest;
        dest.s0 = conn.clToNID;
        ushort modulator = conn.clNeuroMod;
        CLPType cw = conn.clWeight;
        dest.s2 = neuronsSt[dest.s0].clExc;
        //no neuro modulation
        if (modulator == OUT_INDEX) {
            dest.s1 = cw;
        }
        //with neuro modulation
        else {
            CLPType modulatorInput = 0;
            CLNeuron modulatorN = neuronsSt[modulator];
            CLPType neM = modulatorN.clExc;
            //check if it is excited
            if (checkExcitationThreshold(neM)) {
                bool isFired = modulatorN.clFired == true;
                modulatorInput = isFired ? modulatorN.clSt : modulatorN.clPSt;
            }

            dest.s1 = modulatorInput;
        }

        destModCounter++;
        destMod[destModCounter] = dest;
    }

    for (; destModCounter >= 0; destModCounter--) {
        ushort destination = destMod[destModCounter].s0;
        ushort cw = destMod[destModCounter].s1;
        // NOTE!! The line below creates a memory dependency, if it is replaced
        // with the one below, however, without it, calculation of excitation for
        // control neurons is not correct, specially with
//        ushort nExc = neuronsSt[destination].clExc;
        ushort nExc = destMod[destModCounter].s2;
        neuronsSt[destination].clExc = neuronOpr(cw, ns, nExc);
    }
#else
///////////////////////// NOTE //////////////////////////////////
// The code below is needed if code must match what NNetworkState
// does in its function "NNSFunction::process". The problem resides
// in the need to forward excitation values to later executions
// but that creates a data dependency that the FPGA can not handle
// correctly.
////////////////////////////////////////////////////////////////
    CLPType exc = 0;

    // Check the connections from this neuron
    for (ushort i = 0; i < connSize; i++) {
        CLConnection conn = conns[i];
        // Skip if it is not a connection from this Neuron
        if (conn.clFromNID != idx) {
            continue;
        }

        ushort destination = conn.clToNID;
        ushort modulator = conn.clNeuroMod;
        CLPType cw = conn.clWeight;
        CLPType nExc = neuronsSt[destination].clExc;

        //no neuro modulation
        if (modulator == OUT_INDEX) {
            exc = neuronOpr(cw, ns, nExc);
        }
        //with neuro modulation
        else {
            CLPType modulatorInput = 0;
            CLNeuron modulatorN = neuronsSt[modulator];
            CLPType neM = modulatorN.clExc;
            //check if it is excited
            if (checkExcitationThreshold(neM)) {
                bool isFired = modulatorN.clFired == true;
                modulatorInput = isFired ? modulatorN.clSt : modulatorN.clPSt;
            }

            exc = neuronOpr(modulatorInput, ns, nExc);
        }

        neuronsSt[destination].clExc = exc;
    }
////////////////////////////////////////////////////////////////
#endif
}

void processCPrimerNeurons(const ushort2 netSt,
                           global CLNeuron *neurons,
                           global CLConnection *conns,
                           const global CLPType *input,
                           global CLPType *output)
{
    ushort primersRef[ARRAY_MAX];
    int primersCounter = -1;

    for (ushort idx = IO_SIZE; idx < netSt.s0; idx++) {
        if (neurons[idx].clFRNT & 1024) { // ntCPrimer
            primersCounter++;
            primersRef[primersCounter] = idx;
            execute(idx, false, netSt, neurons, conns, input, output);
        }
    }

    for (int i = 0; i <= primersCounter; i++) {
        ushort idx = primersRef[i];
        neurons[idx].clFired = true;
    }

    for (int i = 0; i <= primersCounter; i++) {
        ushort idx = primersRef[i];
        runControlID(idx, netSt.s1, conns, neurons);
    }
}

void processControlNeurons(const ushort2 netSt,
                           global CLNeuron *neurons,
                           global CLConnection *conns,
                           const global CLPType *input,
                           global CLPType *output)
{
    //the loop only stops when:
    //    -no Control Neuron that is excited and was not activated is found
    while (true) {
        ushort activeNeurons[ARRAY_MAX];
        int activeNeuronsCounter = -1;
        //execute all Control Neurons that are excited and not activated
        //        for (ushort idx = mod.neuronsStart; idx < mod.neuronsEnd; idx++) {
        for (ushort idx = IO_SIZE; idx < netSt.s0; idx++) {
            bool isControl = neurons[idx].clFRNT & (256 | 1024); // Contol and CPrimer
            CLPType ne = neurons[idx].clExc;
            bool isExcited = checkExcitationThreshold(ne);
            bool notFired = neurons[idx].clFired == false;
            if (isControl && isExcited && notFired) {
                execute(idx, false, netSt, neurons, conns, input, output);
                activeNeuronsCounter++;
                activeNeurons[activeNeuronsCounter] = idx;
            }
        }

        //stop loop if no excited and not activated control neuron was found
        if (activeNeuronsCounter < 0) {
            return;
        }

        //set all Control Neurons as already activated and fire them
        for (ushort i = 0; i <= activeNeuronsCounter; i++) {
            ushort idx = activeNeurons[i];
            //mark as fired
            neurons[idx].clFired = true;
        }

        //Excite/Inhibit other neurons
        for (; activeNeuronsCounter >= 0; activeNeuronsCounter--) {
            ushort idx = activeNeurons[activeNeuronsCounter];
            runControlID(idx, netSt.s1, conns, neurons);
        }
    }
}

void processRemainingNeurons(const ushort2 netSt,
                             global CLNeuron *neurons,
                             global CLConnection *conns,
                             const global CLPType *input,
                             global CLPType *output)
{
    ushort remainingNeuronsList[ARRAY_MAX];
    int remainingCounter = -1;

    //build the list of remaining neurons
    //    for (ushort idx = mod.neuronsStart; idx < mod.neuronsEnd; idx++) {
    for (ushort idx = INPUT_SIZE; idx < netSt.s0; idx++) {
        ushort nType = neurons[idx].clFRNT; // neuron.clNType;
        CLPType ne = neurons[idx].clExc;
        bool notControl = !(nType & (256 | 1024)); // (ntControl | ntCPrimer));
        bool isExcited = checkExcitationThreshold(ne);
        bool notFired = neurons[idx].clFired == false;

        if (notControl && isExcited && notFired) {
            remainingCounter++;
            remainingNeuronsList[remainingCounter] = idx;
        }
    }

    if (remainingCounter < 0) {
        return;
    }

    //the loop only stops when:
    //    -no Remaining Neuron, that was excited and not activated, is found
    for (;;) {
        ushort2 activeNeurons[ARRAY_MAX];
        int activeCounter = -1;

        for (ushort tempRemainingCounter = 0; tempRemainingCounter <= remainingCounter; tempRemainingCounter++) {
            ushort idx = remainingNeuronsList[tempRemainingCounter];
            CLPType exOut = execute(idx, true, netSt, neurons, conns, input, output);

            if (remainingNeuronThreshold(exOut)) {
                activeCounter++;
                activeNeurons[activeCounter].x = idx;
                activeNeurons[activeCounter].y = tempRemainingCounter;
            }
        }

        //stop loop if no excited and not activated control neuron was found
        if (activeCounter < 0) {
            break;
        }

        //set current active Neurons as already activated and fire them
        for (; activeCounter >= 0; activeCounter--) {
            ushort nID = activeNeurons[activeCounter].x;
            //mark as fired
            neurons[nID].clFired = true;
            ushort posRemaining = activeNeurons[activeCounter].y;
            //remove activated neuron from the list of remaining neurons
            remainingNeuronsList[posRemaining] = remainingNeuronsList[remainingCounter];
            remainingCounter--;
        }
    }
}

kernel void processStateG(global const CLPType * restrict input,
                          global CLPType * restrict output,
                          global CLCell * restrict cells)
{

// The pragma "PIPELINE" signals the kernel it should operate in a
// "single worker" mode. Otherwise it asks OCL for a global ID, which
// is asigned to the variable "i"
#ifdef PIPELINE
    for (ushort i = 0; i < CELL_POPULATION; i++)
#else
// Get identification details for this agent
    size_t i = get_global_id(0);
//    size_t j = get_local_id(0);
//    size_t wg = get_local_size(0);
#endif
    {
        ushort inputOffset = i * INPUT_SIZE;
        ushort outputOffset = i * OUTPUT_SIZE;
        ushort nSize = cells[i].nrsSize;
        ushort cSize = cells[i].connSize;
        
        const ushort2 localNSt = (ushort2)(nSize, cSize);

        const global CLPType *gblInput = &(input[inputOffset]);
        global CLPType *gblOutput = &(output[outputOffset]);

//        if (i == 0) {
//        	printf("gblInput: %i, %i\n", gblInput[0], gblInput[1]);
//        	printCell(cells, 1);
//        } // print once the first cell

        processInputNeurons(localNSt, cells[i].clNrs, cells[i].clCons, gblInput, gblOutput);

        processCPrimerNeurons(localNSt, cells[i].clNrs, cells[i].clCons, gblInput, gblOutput);

        processControlNeurons(localNSt, cells[i].clNrs, cells[i].clCons, gblInput, gblOutput);

        processRemainingNeurons(localNSt, cells[i].clNrs, cells[i].clCons, gblInput, gblOutput);

        processOutputNeurons(localNSt, cells[i].clNrs, cells[i].clCons, gblInput, gblOutput);

        //if (i == 0) { printCell(cells, 1); }

        for (ushort j = 0; j < nSize; j++) {
        	cells[i].clNrs[j].clPSt =  cells[i].clNrs[j].clSt;
            cells[i].clNrs[j].clSt = 0;
            cells[i].clNrs[j].clExc = 0;
            cells[i].clNrs[j].clFired = false;
        }

        for (ushort j = 0; j < cSize; j++) {
            cells[i].clCons[j].clPCnType = cells[i].clCons[j].clCnType;
            cells[i].clCons[j].clCnType = false;
        }

//        if (i == 0) {
//                	printf("gblOutput: %i, %i, %i\n", gblOutput[0], gblOutput[1], gblOutput[2]);
//                	printCell(cells, 1);
//                	printf("***********************\n");
//        } // print once the first cell

    }
}

//FLAGS="-D CELL_POPULATION=100 -D PIPELINE -D NMSIZE=20 -D INITIALBATCH=500 -D CHANCE_OF_CONTROL_NEURON=20 "
//FLAGS+="-D CHANCE_OF_NEUROMODULATION=10 -D CHANCE_OF_ADD_NEURON=2 -D CHANCE_OF_DEL_NEURON=2 -D CHANCE_OF_ADD_CONN=48 "
//FLAGS+="-D CHANCE_OF_DEL_CONN=48 -D CHANCE_OF_WEIGHT_MUT=50 -D STEP_MUTATION=5 -D INPUT_SIZE=2 -D OUTPUT_SIZE=3 "
//FILES=" TycheI.cl CLSupportFunctions.cl CLRandom.cl CLNetState.cl CLNetworkModule.cl CLNMap.cl CLUNM.cl  "

//// // Altera
//aoc -g -board=c5p -report -v -I. $FLAGS -o nsk.aocx $FILES
//
//nohup time aoc -g -board=c5p -report -v -I. $FLAGS -o nsk.aocx AllGlobalState.cl > AlteraCompLog.txt 2>&1 &
//aoc -c -board=c5p -report -v -I. -D CELL_POPULATION=100 -D CONTINUOUS_PARAM -o nsk CLSupportFunctions.cl FPGANetSt.cl
//aoc -c -board=c5p -report -v -I. -D CELL_POPULATION=100 -o nsk CLSupportFunctions.cl FPGANetSt.cl

// Xilinx
// # NOTE: load the settings with command: ~/sourceXilinx.sh
// nohup time xocc -c -R2 -f xilinx_vcu1525_xdma_201830_1 -I. $FLAGS -o nsk.xo AllGlobalState.cl > CompileLog.txt 2>&1 &
// nohup time xocc -c -R2 -f zcu102 -I. $FLAGS -o nsk.xo merged.cl > CompileLog.txt 2>&1 &
// xocc -c -R2 -f xilinx_vcu1525_xdma_201830_1 -I. $FLAGS -o nsk.xo merged.cl $FILES
// xocc -c -R2 -f zcu102 -I. $FLAGS -o nsk.xo $FILES
// xocc -c --platform xilinx_u280_xdma_201910_1 -o nsk.xo
// v++ -c -f xilinx_u280_xdma_201920_1 -o nsk.xo $FLAGS $FILES
// v++ -c -f xilinx_u280_xdma_201910_1 -o nsk.xo $FLAGS merged.cl
// xocc -c --platform xilinx_u280_xdma_201910_1 CLNSK.cl -o nsk.xo
// xocc -t hw -l -g --platform xilinx_u280_xdma_201910_1 -o nsk.xclbin nsk.xo
// xocc -t hw -l -g --platform xilinx_vcu1525_xdma_201830_1 -o nsk.xclbin nsk.xo

// xocc -t hw -l -g --platform xilinx_vcu1525_xdma_201830_1 -o nsk.xclbin nsk.xo
// nohup time xocc -t hw -l -g --platform xilinx_vcu1525_xdma_201830_1 -o nsk.xclbin nsk.xo > LinkLog.txt 2>&1 &
// nohup time xocc -t hw -l -g --platform zcu102 -o nsk.xclbin nsk.xo > LinkLog.txt 2>&1 &
// nohup time v++ -t hw -l -g -f xilinx_u280_xdma_201920_1 -o nsk.xclbin nsk.xo > LinkLog.txt 2>&1 &
