#include "postgres.h"
#include <math.h>
#include "halfvec.h"
#include "unit8vec.h"
#include "utils/builtins.h"

/*
 * Allocate and initialize a new uint8 vector
 */
Uint8Vector *
InitUint8Vector(int dim)
{
	Uint8Vector *result;
	int			size;

	size = UINT8VEC_SIZE(dim);
	result = (Uint8Vector *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = dim;

	return result;
}

/*
 * Convert half vector to uint8 vector with scalar quantization
 */
Uint8Vector *
halfvec_to_uint8vec(HalfVector *halfvec, float min_val, float max_val)
{
	Uint8Vector *result;
	float		scale;
	
	if (max_val <= min_val)
	{
		/* All values are the same */
		result = InitUint8Vector(halfvec->dim);
		for (int i = 0; i < halfvec->dim; i++)
			result->x[i] = 128;  // Mid-point
		return result;
	}
	
	scale = (max_val - min_val) / 255.0f;
	result = InitUint8Vector(halfvec->dim);
	
	for (int i = 0; i < halfvec->dim; i++)
	{
		float		val = HalfToFloat4(halfvec->x[i]);
		float		normalized = (val - min_val) / scale;
		uint8		quantized;
		
		/* Clamp to [0, 255] */
		if (normalized < 0)
			quantized = 0;
		else if (normalized > 255)
			quantized = 255;
		else
			quantized = (uint8)normalized;
		
		result->x[i] = quantized;
	}
	
	return result;
}

/*
 * Convert uint8 vector back to half vector
 */
HalfVector *
uint8vec_to_halfvec(Uint8Vector *uint8vec, float min_val, float max_val)
{
	HalfVector *result;
	float		scale;
	
	scale = (max_val - min_val) / 255.0f;
	result = InitHalfVector(uint8vec->dim);
	
	for (int i = 0; i < uint8vec->dim; i++)
	{
		float		val = min_val + uint8vec->x[i] * scale;
		result->x[i] = Float4ToHalf(val);
	}
	
	return result;
}

/*
 * Find min and max values in a half vector
 */
void
find_min_max_halfvec(HalfVector *vec, float *min_val, float *max_val)
{
	*min_val = HALF_MAX;
	*max_val = -HALF_MAX;
	
	for (int i = 0; i < vec->dim; i++)
	{
		float val = HalfToFloat4(vec->x[i]);
		if (val < *min_val) *min_val = val;
		if (val > *max_val) *max_val = val;
	}
}