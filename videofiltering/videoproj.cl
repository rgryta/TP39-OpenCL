__kernel void videoproj_gaussian(const __global char* in,
                      __global char* out)
{
	uint cols_per_row = 642;

	uint row = get_global_id(0);
	uint col = get_global_id(1);
	 
	uint tl = cols_per_row*row+col;
	uint tm = cols_per_row*row+col;
	uint tr = cols_per_row*row+col;

	uint ml = cols_per_row*(row+1)+(col+1);
	uint mm = cols_per_row*(row+1)+(col+1);
	uint mr = cols_per_row*(row+1)+(col+1);

	uint bl = cols_per_row*(row+2)+(col+2);
	uint bm = cols_per_row*(row+2)+(col+2);
	uint br = cols_per_row*(row+2)+(col+2);

	uint val = (in[tl] + in[tr] + in[bl] + in[br])/16 + 
		(in[tm] + in[ml] + in[mr] + in[bm])/8 +
		(in[mm])/4;

    // Store the result
	out[mm] = val;
};


__kernel void videoproj_sobel(const __global char* in,
                      __global char* out)
{
	uint cols_per_row = 642;

	uint row = get_global_id(0);
	uint col = get_global_id(1);

	uint tl = cols_per_row*row+col;
	uint tm = cols_per_row*row+col;
	uint tr = cols_per_row*row+col;

	uint ml = cols_per_row*(row+1)+(col+1);
	uint mm = cols_per_row*(row+1)+(col+1);
	uint mr = cols_per_row*(row+1)+(col+1);

	uint bl = cols_per_row*(row+2)+(col+2);
	uint bm = cols_per_row*(row+2)+(col+2);
	uint br = cols_per_row*(row+2)+(col+2);

	float Gx = (-1)*in[tl]+in[tr]+(-2)*in[ml]+2*in[mr]+(-1)*in[bl]+in[br];
	float Gy = (-1)*in[tl]+(-2)*in[tm]+(-1)*in[tr]+in[bl]+2*in[bm]+in[br];

	uint val = sqrt((Gx*Gx) + (Gy*Gy));

    // Store the result
	out[640*row+col] = val > 80 ? 0 : in[mm];
};
