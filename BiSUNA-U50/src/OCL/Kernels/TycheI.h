//
//  TycheI.h
//  BiSUNAOpenCL
//
//  Created by RH VT on 16/03/20.
//  Copyright © 2020 R. All rights reserved.
//

#ifndef TycheI_h
#define TycheI_h

#define TYCHE_I_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define TYCHE_I_DOUBLE_MULTI 5.4210108624275221700372640e-20

/**
State of tyche_i RNG.
*/
typedef union __attribute__((aligned(4))) __attribute__((packed))
{
    struct __attribute__((aligned(4))) __attribute__((packed))
    {
        uint a,b,c,d;
    };
    ulong res;
} tycheIState;

void tycheIAdvance(tycheIState *state);
void tycheISeed(tycheIState *state, ulong seed);

/**
Generates a random 64-bit unsigned integer using tyche_i RNG.

@param state State of the RNG to use.
*/
#define tycheIULong(state) (tycheIAdvance(&state), state.res)

/**
Generates a random 32-bit unsigned integer using tyche_i RNG.

@param state State of the RNG to use.
*/
#define tycheIUInt(state) ((uint)tycheIULong(state))
#define tycheIUShort(state) ((ushort)tycheIULong(state))

/**
Generates a random float using tyche_i RNG.

@param state State of the RNG to use.
*/
#define tycheIFloat(state) (tycheIULong(state)*TYCHE_I_FLOAT_MULTI)

/**
Generates a random double using tyche_i RNG.

@param state State of the RNG to use.
*/
#define tycheIDouble(state) (tycheIULong(state)*TYCHE_I_DOUBLE_MULTI)

/**
Generates a random double using tyche_i RNG. Since tyche_i returns 64-bit numbers this is equivalent to tycheIdouble.

@param state State of the RNG to use.
*/
#define tycheIDouble2(state) tycheIDouble(state)

#endif /* TycheI_h */
