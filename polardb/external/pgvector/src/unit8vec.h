#ifndef UINT8VEC_H
#define UINT8VEC_H

#define UINT8VEC_MAX_DIM 16000

#define UINT8VEC_SIZE(_dim)		(offsetof(Uint8Vector, x) + sizeof(uint8)*(_dim))
#define DatumGetUint8Vector(x)	((Uint8Vector *) PG_DETOAST_DATUM(x))
#define PG_GETARG_UINT8VEC_P(x)	DatumGetUint8Vector(PG_GETARG_DATUM(x))
#define PG_RETURN_UINT8VEC_P(x)	PG_RETURN_POINTER(x)

typedef struct Uint8Vector
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int16		dim;			/* number of dimensions */
	int16		unused;			/* reserved for future use, always zero */
	uint8		x[FLEXIBLE_ARRAY_MEMBER];	/* 8-bit unsigned integers */
}			Uint8Vector;

/* 添加min和max值用于反量化 */
typedef struct Uint8VectorMeta
{
	float		min_val;
	float		max_val;
	Uint8Vector vec;
} Uint8VectorMeta;

Uint8Vector *InitUint8Vector(int dim);
Uint8Vector *halfvec_to_uint8vec(HalfVector *halfvec, float min_val, float max_val);
HalfVector *uint8vec_to_halfvec(Uint8Vector *uint8vec, float min_val, float max_val);

void find_min_max_halfvec(HalfVector *vec, float *min_val, float *max_val);

#endif