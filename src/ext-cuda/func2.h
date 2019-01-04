/*
 * Function definition.
 *
 */

// Energy functional contribution (rho x G(rho))
#define FUNCTION (rhop * 0.5 * (1.0 - TANH(xi * (rhop - rhobf))))
